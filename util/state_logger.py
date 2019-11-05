import util.db as db
from config import Config as cfg
import util.vector_builder as vb
import util.util as util


def log_state(p1, p2, meld, run_id):
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

    This function adds the current state and current action.
    The next state (St+1) is inserted in update_state()
    Returns a tuple for each player consisting of their initial state and the row_id of the initial entry into the database for future updates.
    """

    player_1_state_vector = ",".join([str(x) for x in p1['state'].get_player_state(p1['player'])])
    player_1_action_vector = ",".join([str(x) for x in vb.build_card_vector(p1['card'])])

    player_2_state_vector = ",".join([str(x) for x in p2['state'].get_player_state(p2['player'])])
    player_2_action_vector = ",".join([str(x) for x in vb.build_card_vector(p2['card'])])

    winner_state_vector = ",".join([str(x) for x in meld['state'].get_player_state(meld['player'])])
    winner_action = ",".join([str(x) for x in vb.build_meld_cards_vector(meld['mt_list'])])

    with db.open_connection() as conn:
        with conn.cursor() as cursor:
            # Player 1's trick play
            row_id_1 = db.insert_exp(cursor=cursor, agent_id=p1['player'].name, run_id=run_id, vector=player_1_state_vector, action=player_1_action_vector)

            # Player 2's trick play
            row_id_2 = db.insert_exp(cursor=cursor, agent_id=p2['player'].name, run_id=run_id, vector=player_2_state_vector, action=player_2_action_vector)

            # Trick winner's meld play, commented out to simplify process
            # db.insert_exp(cursor, winner.name, run_id, winner_state_vector, winner_action)

        conn.commit()

    return (p1['state'], row_id_1), (p2['state'], row_id_2)


def update_state(p1, p2, winner=None):
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

    p1_reward = util.get_reward(player=p1['player'], state_1=p1['state_1'], state_2=p1['state_2'], winner=winner)
    p2_reward = util.get_reward(player=p2['player'], state_1=p2['state_1'], state_2=p2['state_2'], winner=winner)

    with db.open_connection() as conn:
        with conn.cursor() as cursor:
            db.update_exp(cursor=cursor, next_vector=p1_next_state_vector, reward=p1_reward, row_id=p1['row_id'])
            db.update_exp(cursor=cursor, next_vector=p2_next_state_vector, reward=p2_reward, row_id=p2['row_id'])

        conn.commit()
