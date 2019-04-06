obs = envs.reset()
agents.reset(obs)

'''prepare to start loop'''
if args.test_env:
    test_env_id = int(time.time())
    test_env_step = 0

while True:

    agents.schedule()

    '''interact'''
    while agents.experience_not_enough():

        action = agents.act(
            obs=obs,
            learning_agent_mode='learning',
        )
        
        '''step'''
        obs, reward, done, infos = envs.step(action)
        agents.observe(obs, reward, done, infos,
            learning_agent_mode='learning')

    agents.at_update(learning_agent_mode='learning')
