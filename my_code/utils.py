# Requirments
from gym_trading_env.downloader import download # Crypto info
import datetime
import pandas as pd
import numpy as np
import itertools
import yaml

# Load parameters
with open(r'notebooks\params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


# Load and process price data for multiple cryptocurrency pairs from Binance    
def load_and_process_prices(crypto_pairs, initial_year, initial_month=9, initial_day=1, timeframe='1h', download_data=True):
    """
    Load and process price data for multiple cryptocurrency pairs from Binance.

    Parameters:
    - crypto_pairs: List of cryptocurrency pairs to load data for.
    - initial_year: Initial year for data.
    - initial_month: Initial month for data (default is January).
    - initial_day: Initial day for data (default is 1st).
    - timeframe: Timeframe for the price data (default is '1h').
    - download_data: If True, downloads data from Binance, otherwise loads existing data (default is True).

    Returns:
    - Dictionary of DataFrames with processed price data for each cryptocurrency pair.
    """
    crypto_data = {}

    for pair in crypto_pairs:
        position = pair.find("/")
        filename = f"./data/raw/binance-{pair[:position]}USDT-{timeframe}.pkl"

        if download_data:
            # Download data
            download(exchange_names=["binance"],
                     symbols=[pair],
                     timeframe=timeframe,
                     dir="data/raw",
                     since=datetime.datetime(year=initial_year, month=initial_month, day=initial_day))

        # Check if file exists before loading
        try:
            df = pd.read_pickle(filename)
        except FileNotFoundError:
            print(f"Data file not found for {pair}. Please enable downloading or check the file path.")
            continue

        # Print information about the loaded data
        print(f"{pair} loaded ({df.shape[0]} rows, {df.shape[1]} columns)")

        # Compute additional features
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_8'] = df['close'].rolling(window=8).mean()
        df['SMA_13'] = df['close'].rolling(window=13).mean()

        df['SMA_1_SMA_2_diff'] = abs(df['SMA_5'] - df['SMA_8'])
        df['SMA_1_SMA_3_diff'] = abs(df['SMA_5'] - df['SMA_13'])
        df['SMA_2_SMA_3_diff'] = abs(df['SMA_8'] - df['SMA_13'])
        df['SMA_index'] = df['SMA_1_SMA_2_diff'] + df['SMA_1_SMA_3_diff'] + df['SMA_2_SMA_3_diff']
        df['feature_SMA_index_diff'] = df['SMA_index'].pct_change()

        # Target feature: Price Direction (1 if the price goes up, -1 if the price goes down, testing)
        # df['feature_price_Direction'] = df['close'].shift(-1) - df['close']
        # df['feature_price_Direction'] = df['feature_price_Direction'].apply(lambda x: 1 if x > 0 else -1)
        # df['price_Direction'] = df['feature_price_Direction'].apply(lambda x: 1 if x > 0 else -1)    

        df.dropna(inplace=True)

        # Store in dictionary
        crypto_data[pair] = df

    return crypto_data

# nS & nA generators 
def states_generator(variation_range, n_states):
    min = variation_range[0]
    max = variation_range[1]
    states = [round(min + i * (max - min) / (n_states - 1),6) for i in range(n_states)]
    return states

def portfolio_action_generator_pair(positions):
    availaible_actions = [[a, b] for a in positions for b in positions if a + b <= 1]
    return availaible_actions

def portfolio_action_generator(agent_actions, number_of_currencies):
    availaible_actions = [
        list(prod) for prod in itertools.product(
            agent_actions, repeat=number_of_currencies) if sum(prod) <= 1]
    return availaible_actions

def agent_action_generator(number_of_actions):
    agent_actions = [i/number_of_actions for i in range(number_of_actions)]
    return agent_actions

# Interpreters and transformers
def cord_to_index(cordenates, shape):
    index = np.ravel_multi_index(cordenates, shape) 
    return index

def index_to_cord(index, shape):
    cordenates = np.unravel_index(index, shape)
    return cordenates   

## Interpreter class
class StateInterprete:
    def __init__(self, states_matrix, portfolio, feature_states):
        self.states_matrix = states_matrix
        self.portfolio = portfolio
        self.feature_states = feature_states

    def state_to_index(self, observation):
        num_currencies = len(observation)
        cordenadas = []
        n_features = 1 + 1

        for c in range(num_currencies):
            for i in range(n_features):
                feature = observation[c][i]

                if i == n_features-1: # Estado de los activos del portafolio
                    # print(f'pre_estado: {feature}')  
                    portfolio_positions = self.portfolio.positions
                    feature = min(range(len(portfolio_positions)), key=lambda i: abs(portfolio_positions[i] - feature))
                    # print(f'post_estado: {feature}')
                else:
                    # print(f'pre_feature: {feature}')   
                    feature = min(range(len(self.feature_states)), key=lambda i: abs(self.feature_states[i] - feature))
                    # print(f'post_feature: {feature}')

                cordenadas.append(feature)
      
        indice = np.ravel_multi_index(cordenadas, self.states_matrix)
        return indice
    
    def index_to_state(self, indice):
        cordenadas = np.unravel_index(indice, self.states_matrix)
        return cordenadas

## Visualización de la politica

import itertools

def visual_policy_2f(states_matrix, agent, policy_evolution, episode, feature_states):
    
    matches = []

    # 1M1atriz estados
    states_matrix

    ## Generate all possible states
    all_states = list(itertools.product(*[range(n) for n in states_matrix]))

    ## Create a DataFrame
    df_states = pd.DataFrame(all_states, columns=['F1_BTC', '%BTC', 'F1_SOL', '%SOL'])

    ## Replace values in State1 and State3 columns
    df_states['F1_BTC'] = df_states['F1_BTC'].map(lambda x: feature_states[x])
    df_states['F1_SOL'] = df_states['F1_SOL'].map(lambda x: feature_states[x])

    # Policy
    policy = pd.DataFrame(agent.policy, columns=['No invierte', 'Invierte en BTC', 'Invierte en SOL'])
    df_states['Policy'] = policy[['No invierte', 'Invierte en BTC', 'Invierte en SOL']].idxmax(axis=1)
    df_states['Policy'] = df_states.apply(lambda row: "NA" if (row['%BTC'] == 1 and row['%SOL'] == 1) else row['Policy'], axis=1)
    policy = policy.applymap(lambda x: f"{x*100:.2f}%")

    # Q Values
    q_values = pd.DataFrame(agent.Q, columns=['QA1', 'QA2', 'QA3'])

    visual_policy = pd.concat([df_states,policy,q_values], axis=1)

    if episode == 0:
        policy_evolution = visual_policy.iloc[:, :5].copy()
    else:
        if episode % 5 == 0:
            policy_evolution[f'Policy_{episode}'] = visual_policy["Policy"]

            list_a = ['No invierte','No invierte','Invierte en SOL',
                      'Invierte en SOL','No invierte','NA',
                      'Invierte en SOL','NA','Invierte en BTC',
                      'Invierte en BTC','Invierte en SOL','Invierte en SOL',
                      'Invierte en BTC','NA','Invierte en SOL','NA']
            
            list_b = policy_evolution[f'Policy_{episode}'].tolist()

            matches = sum(1 for a, b in zip(list_a, list_b) if a == b) / 16 * 100

    return visual_policy, policy_evolution, matches

def environment_parameters_generator(crypto_data, agent_actions):

    # Load parameters
    with open(r'notebooks\params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    env_params = []

    for i in range(len(params["crypto_pairs"])):
        env_params.append({
            'name': params["crypto_pairs"][i],
            'df': crypto_data[params["crypto_pairs"][i]],
            'agent_actions': agent_actions,
            'trading_fees': params["trading_fees"],
            'borrow_interest_rate': params["borrow_interest_rate"],
            'initial_position': params["initial_position"],
            'portfolio_initial_value': params["portfolio_initial_value"]
        })

    return env_params