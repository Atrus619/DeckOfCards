import util.db as db
from config import Config as cfg
import util.vector_builder as vb
import util.util as util
import pandas as pd


# TODO: We may want to consider making this part of game.py
def log_state(df, p1, p2, meld, run_id, history):
    """
    PLAYER ORDER: TRICK ORDER

    Player dicts contain the following key/values:
    1. player: player object specifying player
    2. state: initial state before player took action
    3. card: card played by player in trick

    Meld dict contains the following key/values: (NOT CURRENTLY IMPLEMENTED/LIVE)
    1. player: player object specifying player who melded
    2. state: initial state before melding
    3. mt_list: list of cards melded

    This function adds the current state and current action to a specified dataframe.
    The next state (St+1) is inserted in update_state()
    Returns a tuple for each player consisting of their initial state and the row_id of the initial entry into the database for future updates.
    """

    player_1_state_vector = ",".join([str(x) for x in p1['state'].get_player_state(p1['player'])])
    player_1_action_vector = ",".join([str(x) for x in vb.build_card_vector(p1['card'])])

    player_2_state_vector = ",".join([str(x) for x in p2['state'].get_player_state(p2['player'])])
    player_2_action_vector = ",".join([str(x) for x in vb.build_card_vector(p2['card'])])

    # TODO: update logging to include melds
    meld_vector = ",".join([str(x) for x in vb.build_meld_cards_vector(meld["mt_list"])])
    meld_mask = vb.build_meld_mask_vector()

    df2 = pd.DataFrame([
        [p1['player'].name, p2['player'].name, run_id, player_1_state_vector, player_1_action_vector,
         history[p1['player']][2]],

        [p2['player'].name, p1['player'].name, run_id, player_2_state_vector, player_2_action_vector,
         history[p2['player']][2]]
    ],
        columns=['agent_id', 'opponent_id', 'run_id', 'vector', 'action', 'meld_action'])

    df = df.append(df2, sort=False, ignore_index=True)
    df_len = len(df)

    # TODO: revisit to make it sleek possibly, optional
    history[p1['player']] = (p1['state'], df_len - 2,
                             meld_vector if p1["player"] == meld["player"] else meld_mask)
    history[p2['player']] = (p2['state'], df_len - 1,
                             meld_vector if p2["player"] == meld["player"] else meld_mask)

    return df, history


def update_state(df, p1, p2, win_reward, winner=None):
    """
    PLAYER ORDER: TRICK ORDER

    Player dicts contain the following key/values:
    1. player: player object specifying player
    2. state_1: initial state before player took action
    3. state_2: next state observed by player
    4. row_id: row_id of initial entry into database with prior action

    Adds information about the next state (St+1) and reward, see log_state() for additional info

    The first player will have the trick start state only since he is playing the first card
    The second player will have knowledge of what card was played first by first player, hence we need to update
    the state with that information

    Winner is only used if the game is over and it specifies the player object that won the entire game
    """
    if winner is None:
        p1_next_state_vector = ",".join([str(x) for x in p1['state_2'].get_player_state(p1['player'])])
        p2_next_state_vector = ",".join([str(x) for x in p2['state_2'].get_player_state(p2['player'])])
    else:  # Game over, buddy
        p1_next_state_vector = cfg.terminal_state
        p2_next_state_vector = cfg.terminal_state

    p1_reward = util.get_reward(player=p1['player'], state_1=p1['state_1'], state_2=p1['state_2'], winner=winner,
                                win_reward=win_reward)
    p2_reward = util.get_reward(player=p2['player'], state_1=p2['state_1'], state_2=p2['state_2'], winner=winner,
                                win_reward=win_reward)

    df.loc[[p1['row_id'], p2['row_id']], ['next_vector', 'reward']] = [[p1_next_state_vector, p1_reward],
                                                                       [p2_next_state_vector, p2_reward]]
    return df
