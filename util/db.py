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




def insert_exp(cursor, agent_id, run_id, vector, reward, action):
    cursor.execute(
            "INSERT INTO cards.experience \
            (ins_ts, agent_id, run_id, vector, reward, action) \
            VALUES (now(), '" + str(agent_id) + "', '" + str(run_id) + "', '" + str(vector) + "', " + str(reward) + ", '" + str(action) + "');"
            )


def insert_metrics(run_id, win_rate, win_rate_random, average_reward):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO cards.metrics \
                (run_id, win_rate, win_rate_random, average_reward) \
                VALUES ('" + str(run_id) + "', '" + str(win_rate) + "', '" + str(win_rate_random) + "', "
                "'" + str(average_reward) + "');"
            )

        conn.commit()


def get_exp(run_id, buffer):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT vector, reward, action FROM \
                cards.experience \
                WHERE run_id = '" + str(run_id) + "' \
                ORDER BY ins_ts DESC \
                LIMIT " + str(buffer) + ";"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['vector', 'reward', 'action'])
    return df


def get_metrics(run_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT win_rate, win_rate_random, average_reward FROM \
                cards.metrics \
                WHERE run_id = '" + str(run_id) + "' \
                ORDER BY id ASC ;"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['win_rate', 'win_rate_random', 'average_reward'])
    return df


def get_max_id(run_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT max(id)  \
                FROM cards.experience \
                WHERE run_id = '" + str(run_id) + "';"
            )
            result = cursor.fetchone()

    return result[0]


def get_rewards_by_id(run_id, previous_experience_id, agent_id):
    with open_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT reward  \
                FROM cards.experience \
                WHERE run_id = '" + str(run_id) + "' \
                and id > " + str(previous_experience_id) + " \
                and agent_id = '" + str(agent_id) + "' ;"
            )
            result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['reward'])
    return df


