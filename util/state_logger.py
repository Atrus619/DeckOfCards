import util.db as db


def log_state(trick_state, meld_state, card_1, card_2, mt_list, trick_score,
              meld_score, winner, player_1, player_2, run_id):
    score_1 = trick_score if player_1 == winner else 0
    score_2 = trick_score if player_2 == winner else 0

    player_1_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_1)])
    player_1_action_vector = trick_state.build_card_vector(card_1)

    player_2_state_vector = ",".join([str(x) for x in trick_state.get_player_state(player_2)])
    player_2_action_vector = trick_state.build_card_vector(card_2)

    winner_state_vector = ",".join([str(x) for x in meld_state.get_player_state(winner)])
    winner_action = meld_state.build_meld_cards_vector(mt_list)

    with db.open_connection().cursor() as cursor:
        db.insert_exp(cursor, player_1.name, run_id, player_1_state_vector, score_1, player_1_action_vector)

        db.insert_exp(cursor, player_2.name, run_id, player_2_state_vector, score_2, player_2_action_vector)

        db.insert_exp(cursor, winner.name, run_id, winner_state_vector, meld_score, winner_action)
