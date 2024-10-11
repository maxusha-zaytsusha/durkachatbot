import sqlite3

# скрипт для создания пустой базы настроек чатов
conn = sqlite3.connect("base.sql")
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS users (id int primary key, model int, output_method int)")
cur.commit()
cur.close()
conn.cloase()