from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3

app = Flask(__name__)
app.secret_key = 'some_secret'

# Initialize the database connection
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_name TEXT NOT NULL,
                    date TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('Database_viewer.html')

@app.route('/execute_query', methods=['POST'])
def execute_query():
    query = request.form.get('query')
    try:
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute(query)
        conn.commit()
        
        if query.lower().startswith("select"):
            rows = c.fetchall()
            columns = [description[0] for description in c.description]
            return render_template('Database_viewer.html', columns=columns, rows=rows, query=query)
        else:
            flash('Query executed successfully', 'success')
            return redirect(url_for('index'))
    except sqlite3.Error as e:
        flash(f'An error occurred: {e}', 'danger')
        return redirect(url_for('index'))
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True,port=8080)
