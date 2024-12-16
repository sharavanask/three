import React, { useState } from 'react';
import axios from 'axios';

const AddEmployeeForm = () => {
    const [formData, setFormData] = useState({
        name: '',
        employee_id: '',
        email: '',
        phone_number: '',
        department: '',
        date_of_joining: '',
        role: '',
    });

    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage('');
        setError('');

        try {
            const response = await axios.post('http://localhost:5000/add-employee', formData);
            setMessage(response.data.message);
            setFormData({
                name: '',
                employee_id: '',
                email: '',
                phone_number: '',
                department: '',
                date_of_joining: '',
                role: '',
            });
        } catch (err) {
            setError(err.response?.data?.message || 'An error occurred.');
        }
    };

    return (
        <div className="container mt-5">
            <h2>Add Employee</h2>
            {message && <div className="alert alert-success">{message}</div>}
            {error && <div className="alert alert-danger">{error}</div>}
            <form onSubmit={handleSubmit}>
                <div className="mb-3">
                    <label>Name</label>
                    <input
                        type="text"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        className="form-control"
                        required
                    />
                </div>
                <div className="mb-3">
                    <label>Employee ID</label>
                    <input
                        type="text"
                        name="employee_id"
                        value={formData.employee_id}
                        onChange={handleChange}
                        className="form-control"
                        maxLength="10"
                        required
                    />
                </div>
                <div className="mb-3">
                    <label>Email</label>
                    <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        className="form-control"
                        required
                    />
                </div>
                <div className="mb-3">
                    <label>Phone Number</label>
                    <input
                        type="text"
                        name="phone_number"
                        value={formData.phone_number}
                        onChange={handleChange}
                        className="form-control"
                        pattern="\d{10}"
                        title="Enter a valid 10-digit phone number"
                        required
                    />
                </div>
                <div className="mb-3">
                    <label>Department</label>
                    <select
                        name="department"
                        value={formData.department}
                        onChange={handleChange}
                        className="form-control"
                        required
                    >
                        <option value="">Select Department</option>
                        <option value="HR">HR</option>
                        <option value="Engineering">Engineering</option>
                        <option value="Marketing">Marketing</option>
                    </select>
                </div>
                <div className="mb-3">
                    <label>Date of Joining</label>
                    <input
                        type="date"
                        name="date_of_joining"
                        value={formData.date_of_joining}
                        onChange={handleChange}
                        className="form-control"
                        max={new Date().toISOString().split('T')[0]}
                        required
                    />
                </div>
                <div className="mb-3">
                    <label>Role</label>
                    <input
                        type="text"
                        name="role"
                        value={formData.role}
                        onChange={handleChange}
                        className="form-control"
                        required
                    />
                </div>
                <button type="submit" className="btn btn-primary">
                    Submit
                </button>
                <button
                    type="reset"
                    className="btn btn-secondary ms-2"
                    onClick={() => setFormData({
                        name: '',
                        employee_id: '',
                        email: '',
                        phone_number: '',
                        department: '',
                        date_of_joining: '',
                        role: '',
                    })}
                >
                    Reset
                </button>
            </form>
        </div>
    );
};

export default AddEmployeeForm;
