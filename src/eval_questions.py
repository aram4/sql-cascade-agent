"""Eval questions for the sample database with gold SQL answers."""

SAMPLE_EVAL = [
    {
        "question_id": 0,
        "question": "How many employees are there?",
        "gold_sql": "SELECT COUNT(*) FROM employees",
    },
    {
        "question_id": 1,
        "question": "What is the total budget across all departments?",
        "gold_sql": "SELECT SUM(budget) FROM departments",
    },
    {
        "question_id": 2,
        "question": "Which employees work in the Engineering department?",
        "gold_sql": "SELECT e.name FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = 'Engineering'",
    },
    {
        "question_id": 3,
        "question": "What is the average salary of employees hired after 2021?",
        "gold_sql": "SELECT AVG(salary) FROM employees WHERE hire_date > '2021-12-31'",
    },
    {
        "question_id": 4,
        "question": "Which department has the most employees?",
        "gold_sql": "SELECT d.dept_name FROM departments d JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_name ORDER BY COUNT(*) DESC LIMIT 1",
    },
    {
        "question_id": 5,
        "question": "List all projects that have not ended yet.",
        "gold_sql": "SELECT project_name FROM projects WHERE end_date IS NULL",
    },
    {
        "question_id": 6,
        "question": "Who is assigned as Lead on the Cloud Migration project?",
        "gold_sql": "SELECT e.name FROM employees e JOIN assignments a ON e.emp_id = a.emp_id JOIN projects p ON a.project_id = p.project_id WHERE p.project_name = 'Cloud Migration' AND a.role = 'Lead'",
    },
    {
        "question_id": 7,
        "question": "How many projects does Alice Chen work on?",
        "gold_sql": "SELECT COUNT(*) FROM assignments a JOIN employees e ON a.emp_id = e.emp_id WHERE e.name = 'Alice Chen'",
    },
    {
        "question_id": 8,
        "question": "What is the highest salary in the company?",
        "gold_sql": "SELECT MAX(salary) FROM employees",
    },
    {
        "question_id": 9,
        "question": "Which employees earn more than the average salary?",
        "gold_sql": "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    },
    {
        "question_id": 10,
        "question": "List departments with a budget over 200000.",
        "gold_sql": "SELECT dept_name FROM departments WHERE budget > 200000",
    },
    {
        "question_id": 11,
        "question": "How many employees does the Sales department have?",
        "gold_sql": "SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = 'Sales'",
    },
    {
        "question_id": 12,
        "question": "Which projects started in 2023 and belong to the Engineering department?",
        "gold_sql": "SELECT p.project_name FROM projects p JOIN departments d ON p.dept_id = d.dept_id WHERE p.start_date >= '2023-01-01' AND p.start_date < '2024-01-01' AND d.dept_name = 'Engineering'",
    },
    {
        "question_id": 13,
        "question": "What is the name and salary of the lowest paid employee?",
        "gold_sql": "SELECT name, salary FROM employees ORDER BY salary ASC LIMIT 1",
    },
    {
        "question_id": 14,
        "question": "How many employees are assigned to more than one project?",
        "gold_sql": "SELECT COUNT(*) FROM (SELECT emp_id FROM assignments GROUP BY emp_id HAVING COUNT(*) > 1)",
    },
]
