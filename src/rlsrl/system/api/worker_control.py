from typing import Dict, List, Tuple, Union
import collections
import dataclasses
import multiprocessing as mp
import multiprocessing.connection as mp_connection

from rlsrl.base.shared_memory import (OutOfOrderSharedMemoryControl,
                                      SharedMemoryInferenceStreamCtrl,
                                      PinnedRequestSharedMemoryControl,
                                      PinnedResponseSharedMemoryControl)
import rlsrl.api.config as config_api


@dataclasses.dataclass
class WorkerCtrl:
    rx: mp_connection.Connection
    inf_ctrls: Tuple[SharedMemoryInferenceStreamCtrl] = None
    spl_ctrls: Tuple[OutOfOrderSharedMemoryControl] = None


def _reveal_shm_stream_identity(setup: config_api.ExperimentConfig):
    inf_streams = set()
    spl_streams = set()
    spl_specs = dict()

    # Collect all shared memory stream names.
    for aw in setup.actors:
        for i, x in enumerate(aw.inference_streams):
            if (not isinstance(x, str) and x.type_
                    == config_api.InferenceStream.Type.SHARED_MEMORY):
                inf_streams.add(aw.inference_streams[i].stream_name)

        for i, x in enumerate(aw.sample_streams):
            if (not isinstance(x, str)
                    and x.type_ == config_api.SampleStream.Type.SHARED_MEMORY):
                stream_name = aw.sample_streams[i].stream_name
                spl_streams.add(stream_name)
                if stream_name not in spl_specs:
                    spl_specs[stream_name] = (aw.sample_streams[i].qsize,
                                              aw.sample_streams[i].reuses,
                                              aw.sample_streams[i].batch_size)
                else:
                    assert spl_specs[stream_name] == (
                        aw.sample_streams[i].qsize,
                        aw.sample_streams[i].reuses,
                        aw.sample_streams[i].batch_size
                    ), ("Inconsistent shared memory stream specification. "
                        "Specs like reuses and qsize should be the same with the same stream name."
                        )

    return setup, sorted(inf_streams), sorted(spl_streams), spl_specs


def make_worker_control(experiment_name: str, trial_name: str,
                        setup: config_api.ExperimentConfig):
    # Make worker control.
    (setup, inf_streams, spl_streams,
     spl_specs) = _reveal_shm_stream_identity(setup)

    inf_ctrls = [
        SharedMemoryInferenceStreamCtrl(
            request_ctrl=PinnedRequestSharedMemoryControl(
                experiment_name,
                trial_name,
                f"{x}_infreq_ctrl",
            ),
            response_ctrl=PinnedResponseSharedMemoryControl(
                experiment_name,
                trial_name,
                f"{x}_infresp_ctrl",
            )) for x in inf_streams
    ]
    spl_ctrls = [
        OutOfOrderSharedMemoryControl(
            experiment_name,
            trial_name,
            f"{x}_spl_ctrl",
            qsize=spl_specs[x][0],
            reuses=spl_specs[x][1],
        ) for x in spl_streams
    ]

    tx_handles = collections.defaultdict(list)
    ctrls = collections.defaultdict(list)

    # Assign shared memory stream index to each worker.
    # Stream ctrls will be indexed correspondingly.
    for aw in setup.actors:
        for i, x in enumerate(aw.inference_streams):
            if (not isinstance(x, str) and x.type_
                    == config_api.InferenceStream.Type.SHARED_MEMORY):
                x.stream_index = inf_streams.index(x.stream_name)
        for i, x in enumerate(aw.sample_streams):
            if (not isinstance(x, str)
                    and x.type_ == config_api.SampleStream.Type.SHARED_MEMORY):
                x.stream_index = spl_streams.index(x.stream_name)
        tx, rx = mp.Pipe()
        ctrls['actor'].append(WorkerCtrl(rx, inf_ctrls, spl_ctrls))
        tx_handles['actor'].append(tx)

    for pw in setup.policies:
        if (not isinstance(pw.inference_stream, str)
                and pw.inference_stream.type_
                == config_api.InferenceStream.Type.SHARED_MEMORY):
            pw.inference_stream.stream_index = inf_streams.index(
                pw.inference_stream.stream_name)
        tx, rx = mp.Pipe()
        ctrls['policy'].append(WorkerCtrl(rx, inf_ctrls, None))
        tx_handles['policy'].append(tx)

    for tw in setup.trainers:
        if (not isinstance(tw.sample_stream, str) and tw.sample_stream.type_
                == config_api.SampleStream.Type.SHARED_MEMORY):
            tw.sample_stream.stream_index = spl_streams.index(
                tw.sample_stream.stream_name)
        tx, rx = mp.Pipe()
        ctrls['trainer'].append(WorkerCtrl(rx, None, spl_ctrls))
        tx_handles['trainer'].append(tx)

    for em in setup.eval_managers:
        if (not isinstance(em.eval_sample_stream, str)
                and em.eval_sample_stream.type_
                == config_api.SampleStream.Type.SHARED_MEMORY):
            em.eval_sample_stream.stream_index = spl_streams.index(
                em.eval_sample_stream.stream_name)
        tx, rx = mp.Pipe()
        ctrls['eval_manager'].append(WorkerCtrl(rx, None, spl_ctrls))
        tx_handles['eval_manager'].append(tx)

    return setup, dict(ctrls), dict(tx_handles)