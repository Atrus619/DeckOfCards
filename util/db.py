import psycopg2 as pg
import psycopg2.extras as pge
import pandas as pd


def open_connection():
    connection = pg.connect(
        database="postgres",
        user="postgres",
        host="localhost",
        password="password"
    )

    return connection


def upload_exp(df):
    if len(df) > 0:
        columns = ",".join(df.columns)

        values = "VALUES ({})".format(",".join(["%s" for _ in df.columns]))

        insert_stmt = "INSERT INTO {} ({}) {}".format('cards.experience', columns, values)

        with open_connection() as conn:
            with conn.cursor() as cursor:
                pge.execute_batch(cursor, insert_stmt, df.values)
            conn.commit()


def archive_exp(df):
    if len(df) > 0:
        columns = ",".join(df.columns)

        values = "VALUES ({})".format(",".join(["%s" for _ in df.columns]))

        insert_stmt = "INSERT INTO {} ({}) {}".format('cards.experience_archive', columns, values)

        with open_connection() as conn:
            with conn.cursor() as cursor:
                pge.execute_batch(cursor, insert_stmt, df.values)
            conn.commit()


def get_all_exp():
    """
    Select order is different from other queries
    """
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT id, ins_ts, agent_id, run_id, vector, reward, action, next_vector FROM \
                cards.experience \
                ORDER BY ins_ts DESC;"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['id', 'ins_ts', 'agent_id', 'run_id', 'vector', 'reward', 'action', 'next_vector'])
    return df


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


def get_rewards_by_id(run_id, previous_experience_id, agent_id, opponent_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT reward  \
                FROM cards.experience \
                WHERE run_id = '{run_id}' \
                and id > {previous_experience_id} \
                and agent_id = '{agent_id}' \
                and opponent_id = '{opponent_id}';"
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
