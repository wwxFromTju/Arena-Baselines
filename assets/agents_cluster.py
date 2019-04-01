import torch

class MultiAgentCluster(object):
    """docstring for MultiAgentCluster."""
    def __init__(self, agents, learning_agent_id):
        super(MultiAgentCluster, self).__init__()
        self.all_agents = agents
        self.learning_agent_id = learning_agent_id
        self.learning_agent = self.all_agents[self.learning_agent_id]
        self.playing_agents = self.all_agents[:self.learning_agent_id]+self.all_agents[self.learning_agent_id+1:]
        self.num_agents = len(self.all_agents)

    def reset(self, obs):
        for agent in self.all_agents:
            agent.reset(obs[:,agent.id])

    def schedule(self):
        self.learning_agent.schedule_trainer()

    def experience_not_enough(self):
        return self.learning_agent.experience_not_enough()

    def act(self, obs, learning_agent_mode):
        action = []
        for agent in self.all_agents:
            action += [agent.act(
                obs = obs[:,agent.id],
                mode = learning_agent_mode if agent.id==self.learning_agent_id else 'playing',
            ).unsqueeze(1)]
        return torch.cat(action,1)

    def observe(self, obs, reward, done, infos, learning_agent_mode):
        for agent in self.all_agents:
            agent.observe(
                obs[:,agent.id], reward[:,agent.id], done[:,agent.id], infos,
                mode = learning_agent_mode if agent.id==self.learning_agent_id else 'playing',
            )

    def at_update(self,learning_agent_mode):
        for agent in self.all_agents:
            agent.at_update(
                mode = learning_agent_mode if agent.id==self.learning_agent_id else 'playing',
            )

    def restore_playing_agents(self, principle):
        for agent in self.playing_agents:
            agent.restore(principle=principle)
