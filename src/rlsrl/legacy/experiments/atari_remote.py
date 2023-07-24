import dataclasses
import itertools
import functools

from rlsrl.api.config import *


@dataclasses.dataclass
class AtariMultiNodeExperiment(Experiment):
    aws: int = 32
    pws: int = 4
    tws: int = 1

    inference_splits: int = 4
    ring_size: int = 40

    seed: int = 1

    def make_envs(self, aw_rank):
        return [
            Environment(type_="atari",
                        args=dict(game_name="PongNoFrameskip-v4",
                                  seed=self.seed + 12345 * x +
                                  aw_rank * self.ring_size,
                                  render=False,
                                  pause=False,
                                  noop_max=30,
                                  frame_skip=4,
                                  stacked_observations=4,
                                  max_episode_steps=108000,
                                  gray_scale=True,
                                  obs_shape=(84, 84)))
            for x in range(self.ring_size)
        ]

    def initial_setup(self):
        sample_stream_qsize = 1024

        self.policy_name = "default"
        self.policy = Policy(
            type_="actor-critic",
            args=dict(
                obs_dim={"obs": (4, 84, 84)},
                action_dim=6,
                num_dense_layers=0,
                hidden_dim=512,
                popart=True,
                layernorm=False,
                shared_backbone=True,
                rnn_type='lstm',
                num_rnn_layers=0,
                seed=self.seed,
                cnn_layers=dict(obs=[(16, 8, 4, 0,
                                      'zeros'), (32, 4, 2, 0, 'zeros')]),
                chunk_len=10,
            ))
        self.trainer = Trainer(type_="mappo",
                               args=dict(
                                   discount_rate=0.99,
                                   gae_lambda=0.97,
                                   eps_clip=0.2,
                                   clip_value=True,
                                   dual_clip=False,
                                   vtrace=False,
                                   value_loss='huber',
                                   value_loss_weight=1.0,
                                   value_loss_config=dict(delta=10.0, ),
                                   entropy_bonus_weight=0.01,
                                   optimizer='adam',
                                   optimizer_config=dict(lr=5e-4),
                                   popart=True,
                                   max_grad_norm=40.0,
                                   bootstrap_steps=1,
                                   recompute_adv_among_epochs=False,
                                   recompute_adv_on_reuse=False,
                                   burn_in_steps=0,
                               ))
        self.agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                send_full_trajectory=False,
                send_after_done=False,
                sample_steps=50,
                bootstrap_steps=1,
            )
        ]

        spl_stream = SampleStream(
            type_=SampleStream.Type.NAME,
            stream_name=self.policy_name,
        )
        inf_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                     stream_name=self.policy_name)

        actors = [
            ActorWorker(env=self.make_envs(i),
                        inference_streams=[inf_stream],
                        sample_streams=[spl_stream],
                        agent_specs=self.agent_specs,
                        max_num_steps=108000,
                        inference_splits=self.inference_splits,
                        ring_size=self.ring_size,
                        worker_info=WorkerInformation(node_rank=1))
            for i in range(self.aws)
        ]

        policies = [
            PolicyWorker(
                policy_name=self.policy_name,
                inference_stream=inf_stream,
                policy=self.policy,
                pull_frequency_seconds=None,
                parameter_service_client=ParameterServiceClient(),
                worker_info=WorkerInformation(device="cuda:0", node_rank=0),
            ) for i in range(self.pws)
        ]

        return ExperimentConfig(
            actors=actors,
            policies=policies,
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=False,
                    buffer_args=dict(
                        max_size=sample_stream_qsize,
                        reuses=1,
                        batch_size=32,
                    ),
                    policy_name=self.policy_name,
                    trainer=self.trainer,
                    policy=self.policy,
                    log_frequency_seconds=5,
                    sample_stream=spl_stream,
                    worker_info=WorkerInformation(
                        wandb_job_type='tw',
                        wandb_group='mini',
                        wandb_project='srl-atari',
                        wandb_name=f'seed{self.seed}',
                        log_terminal=True,
                        device="cuda:0",
                        node_rank=0,
                    ),
                ) for _ in range(self.tws)
            ],
            master_worker=[MasterWorker(
                address="10.210.12.42",
                port=51234,
            )])


register_experiment("atari-mini-multinode",
                    functools.partial(AtariMultiNodeExperiment))
