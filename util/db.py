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
