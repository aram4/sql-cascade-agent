"""Create a sample SQLite database for testing the agent before plugging in BIRD."""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sample", "sample.sqlite")


def create():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            budget REAL
        );

        CREATE TABLE IF NOT EXISTS employees (
            emp_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER REFERENCES departments(dept_id),
            salary REAL,
            hire_date TEXT
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            dept_id INTEGER REFERENCES departments(dept_id),
            start_date TEXT,
            end_date TEXT
        );

        CREATE TABLE IF NOT EXISTS assignments (
            emp_id INTEGER REFERENCES employees(emp_id),
            project_id INTEGER REFERENCES projects(project_id),
            role TEXT,
            PRIMARY KEY (emp_id, project_id)
        );

        INSERT OR IGNORE INTO departments VALUES
            (1, 'Engineering', 500000),
            (2, 'Marketing', 200000),
            (3, 'Sales', 300000),
            (4, 'HR', 150000);

        INSERT OR IGNORE INTO employees VALUES
            (1, 'Alice Chen', 1, 120000, '2020-03-15'),
            (2, 'Bob Smith', 1, 95000, '2021-06-01'),
            (3, 'Carol Davis', 2, 85000, '2019-11-20'),
            (4, 'Dan Lee', 3, 90000, '2022-01-10'),
            (5, 'Eve Martinez', 1, 110000, '2020-08-05'),
            (6, 'Frank Wilson', 3, 88000, '2021-03-22'),
            (7, 'Grace Kim', 4, 75000, '2023-02-14'),
            (8, 'Hank Brown', 2, 92000, '2020-05-30');

        INSERT OR IGNORE INTO projects VALUES
            (1, 'Cloud Migration', 1, '2023-01-01', '2023-12-31'),
            (2, 'Brand Refresh', 2, '2023-03-01', '2023-09-30'),
            (3, 'Sales Portal', 3, '2023-06-01', NULL),
            (4, 'ML Pipeline', 1, '2023-04-01', '2024-03-31');

        INSERT OR IGNORE INTO assignments VALUES
            (1, 1, 'Lead'),
            (2, 1, 'Developer'),
            (5, 1, 'Developer'),
            (1, 4, 'Lead'),
            (3, 2, 'Lead'),
            (8, 2, 'Designer'),
            (4, 3, 'Lead'),
            (6, 3, 'Sales Rep');
    """)

    conn.commit()
    conn.close()
    print(f"Sample database created at {DB_PATH}")


if __name__ == "__main__":
    create()
