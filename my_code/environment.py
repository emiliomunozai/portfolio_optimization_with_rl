# Import the necessary libraries
import gymnasium as gym
import gym_trading_env
from my_code.utils import states_generator, portfolio_action_generator, agent_action_generator

# Define the Portfolio class
class Portfolio:
    # Initialize the class with a list of environments
    def __init__(self, env_params, debt_exposure, number_of_currencies):
        self.envs = [self.create_env(params) for params in env_params]
        self.debt_exposure = debt_exposure
        self.positions = env_params[0]['agent_actions']
        self.portfolio_actions = portfolio_action_generator(env_params[0]['agent_actions'],number_of_currencies) # Volver parametrico

    # Make each environment
    def create_env(self, params):
        return gym.make("TradingEnv",
                        name=params['name'],
                        df=params['df'],
                        positions=params['agent_actions'],
                        trading_fees=params['trading_fees'],
                        borrow_interest_rate=params['borrow_interest_rate'],
                        initial_position=params['initial_position'],
                        portfolio_initial_value=params['portfolio_initial_value']
                        )
    
    # Reset all the environments and return their initial states
    def reset(self):
        states = []
        # Dado un estado me devuelve el indice
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return states

    # Take a list of actions (one for each environment) and apply them to the corresponding environments
    # Return the next states, rewards, dones and infos for each environment
    def step(self, portfolio_action_index):
        portfolio_action = self.portfolio_actions[portfolio_action_index]
        #print(f"Equivale a accion del portfolio: {portfolio_action}")
        
        next_states = []
        rewards = []
        dones = []
        infos = []
        extra_values = [] # Create a new list to store the extra values
        
        for i, env in enumerate(self.envs):
            # Get the action index from the agent_actions
            action_index = self.positions.index(portfolio_action[i])
            # Match portfolio action with agent action index
            next_state, reward, done, info, extra_value = self.envs[i].step(action_index)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            extra_values.append(extra_value) # Append the extra value to the list
        # Return the extra values along with the other values
        return next_states, rewards, dones, infos, extra_values