import psycopg2 as pg


def open_connection():
    connection = pg.connect(
        database="postgres",
        user="postgres",
        host="localhost",
        password="password"
    )

    return connection


def get_exp():
    connection = open_connection()

    cursor = connection.cursor()

    cursor.execute("""SELECT * from cards.experience""")

    rows = cursor.fetchall()

    print(rows)

    for row in rows:
        print("   ", row[0])

    connection.close()
