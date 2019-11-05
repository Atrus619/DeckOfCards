import psycopg2 as pg
import pandas as pd


def open_connection():
    connection = pg.connect(
        database="postgres",
        user="postgres",
        host="localhost",
        password="password"
    )

    return connection


def insert_exp(cursor, agent_id, run_id, vector, action):
    """Updates db with information about current state and action"""
    cursor.execute(
            f"INSERT INTO cards.experience \
            (ins_ts, agent_id, run_id, vector, action) \
            VALUES (now(), '{agent_id}', '{run_id}', '{vector}', '{action}') \
            RETURNING id;"
            )

    result = cursor.fetchone()
    return result[0]


def update_exp(cursor, next_vector, reward, row_id):
    """Re-updates db with information about next_state and resulting reward"""
    cursor.execute(
            f"UPDATE cards.experience \
            SET ins_ts = now(), next_vector = '{next_vector}', reward = '{reward}' \
            WHERE id = '{row_id}';"
            )


def insert_metrics(run_id, win_rate, win_rate_random, win_rate_expert_policy, average_reward):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO cards.metrics \
                (run_id, win_rate, win_rate_random, win_rate_expert_policy, average_reward) \
                VALUES ('{run_id}', '{win_rate}', '{win_rate_random}', '{win_rate_expert_policy}', '{average_reward}');"
            )

        conn.commit()


def get_exp(run_id, buffer):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT vector, action, next_vector, reward FROM \
                cards.experience \
                WHERE run_id = '{run_id}' \
                ORDER BY ins_ts DESC \
                LIMIT {buffer};"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['state', 'action', 'next_state', 'reward'])
    return df


def get_metrics(run_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT win_rate, win_rate_random, win_rate_expert_policy, average_reward FROM \
                cards.metrics \
                WHERE run_id = '{run_id}' \
                ORDER BY id ASC ;"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['win_rate', 'win_rate_random', 'win_rate_expert_policy', 'average_reward'])
    return df


def get_max_id(run_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT max(id)  \
                FROM cards.experience \
                WHERE run_id = '{run_id}';"
            )
            result = cursor.fetchone()

    return result[0]


def get_global_max_id():
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT max(id)  \
                FROM cards.experience;"
            )
            result = cursor.fetchone()

    return result[0]


def get_rewards_by_id(run_id, previous_experience_id, agent_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT reward  \
                FROM cards.experience \
                WHERE run_id = '{run_id}' \
                and id > {previous_experience_id} \
                and agent_id = '{agent_id}' ;"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['reward'])
    return df


def clear_run(run_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM cards.experience \
                WHERE run_id = '{run_id}';"
            )
            cursor.execute(
                f"DELETE FROM cards.metrics \
                WHERE run_id = '{run_id}';"
            )

        conn.commit()
