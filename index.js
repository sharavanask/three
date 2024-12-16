const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const mysql = require('mysql2');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// MySQL connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'Sharak@2004', // Replace with your MySQL password
    database: 'employee_db2',
});

db.connect((err) => {
    if (err) {
        console.error('Database connection failed:', err);
        return;
    }
    console.log('Connected to MySQL database.');
});

// Add Employee API
app.post('/add-employee', (req, res) => {
    const { name, employee_id, email, phone_number, department, date_of_joining, role } = req.body;

    if (!name || !employee_id || !email || !phone_number || !department || !date_of_joining || !role) {
        return res.status(400).json({ message: 'All fields are mandatory.' });
    }

    const query = `INSERT INTO employees (name, employee_id, email, phone_number, department, date_of_joining, role) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)`;

    db.query(query, [name, employee_id, email, phone_number, department, date_of_joining, role], (err) => {
        if (err) {
            if (err.code === 'ER_DUP_ENTRY') {
                return res.status(400).json({ message: 'Employee ID or Email already exists.' });
            }
            return res.status(500).json({ message: 'Database error.', error: err });
        }
        res.status(201).json({ message: 'Employee added successfully.' });
    });
});

// Start server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
