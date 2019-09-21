import psycopg2 as pg


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
            VALUES(now(), " + agent_id + ", " + run_id + ", " + vector + ", " + reward + ", " + action + ");"
            )



def get_exp():
    connection = open_connection()

    cursor = connection.cursor()

    cursor.execute("""SELECT * from cards.experience""")

    rows = cursor.fetchall()

    print(rows)

    for row in rows:
        print("   ", row[0])

    connection.close()
