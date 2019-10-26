import util.db as db
from config import Config as cfg
import util.vector_builder as vb
import util.util as util


def log_state(trick_start_state, first_move_state, meld_state, card_1, card_2, mt_list, trick_score,
              meld_score, winner, player_1, player_2, run_id):
    """
    PLAYER ORDER: GAME ORDER

    This function adds the current state and current action.
    The next state (ST+1) is inserted in update_state()
    """

    score_1 = util.get_trick_reward(trick_score=trick_score, player=player_1, winner=winner)
    score_2 = util.get_trick_reward(trick_score=trick_score, player=player_2, winner=winner)

    player_1_state_vector = ",".join([str(x) for x in trick_start_state.get_player_state(player_1)])
    player_1_action_vector = ",".join([str(x) for x in vb.build_card_vector(card_1)])

    player_2_state_vector = ",".join([str(x) for x in first_move_state.get_player_state(player_2)])
    player_2_action_vector = ",".join([str(x) for x in vb.build_card_vector(card_2)])

    winner_state_vector = ",".join([str(x) for x in meld_state.get_player_state(winner)])
    winner_action = ",".join([str(x) for x in vb.build_meld_cards_vector(mt_list)])

    with db.open_connection() as conn:
        with conn.cursor() as cursor:
            # Player 1's trick play
            row_id_1 = db.insert_exp(cursor, player_1.name, run_id, player_1_state_vector, score_1, player_1_action_vector)

            # Player 2's trick play
            row_id_2 = db.insert_exp(cursor, player_2.name, run_id, player_2_state_vector, score_2, player_2_action_vector)

            # Trick winner's meld play, commented out to simplify process
            # db.insert_exp(cursor, winner.name, run_id, winner_state_vector, meld_score, winner_action)

        conn.commit()

    return row_id_1, row_id_2


def update_state(trick_start_state, first_move_state, player_1, player_2, row_id_1, row_id_2):
    """
    PLAYER ORDER: TRICK ORDER

    This function adds the next state (ST+1), see log_state() for additional info

    The first player will have the trick start state only since he is playing the first card
    The second player will have knowledge of what card was played first by first player, hence we need to update
    the state with that information
    """

    player_1_next_state_vector = ",".join([str(x) for x in trick_start_state.get_player_state(player_1)])
    player_2_next_state_vector = ",".join([str(x) for x in first_move_state.get_player_state(player_2)])

    with db.open_connection() as conn:
        with conn.cursor() as cursor:
            db.update_exp(cursor, player_1_next_state_vector, row_id_1)
            db.update_exp(cursor, player_2_next_state_vector, row_id_2)

        conn.commit()


def set_terminal_state(row_id_1, row_id_2):

    with db.open_connection() as conn:
        with conn.cursor() as cursor:
            db.update_exp(cursor, cfg.terminal_state, row_id_1)
            db.update_exp(cursor, cfg.terminal_state, row_id_2)

        conn.commit()
# EAT YOUR EMPTY LINE
