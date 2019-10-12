import util.db as db
from config import Config as cfg


def log_state(trick_state, meld_state, card_1, card_2, mt_list, trick_score,
              meld_score, winner, player_1, player_2, run_id):

    score_1 = trick_score if player_1 == winner else 0
    score_2 = trick_score if player_2 == winner else 0

    player_1_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_1)])
    player_1_action_vector = ",".join([str(x) for x in trick_state.build_card_vector(card_1)])

    player_2_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_2)])
    player_2_action_vector = ",".join([str(x) for x in trick_state.build_card_vector(card_2)])

    winner_state_vector = ",".join([str(x) for x in meld_state.get_player_state(winner)])
    winner_action = ",".join([str(x) for x in meld_state.build_meld_cards_vector(mt_list)])

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


def update_state(trick_state, player_1, player_2, row_id_1, row_id_2):

    player_1_next_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_1)])
    player_2_next_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_2)])

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
