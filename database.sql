-- 1. Users (no foreign keys)
CREATE TABLE Users (
user_id INT AUTO_INCREMENT PRIMARY KEY,
username VARCHAR(50) NOT NULL UNIQUE,
email VARCHAR(100) NOT NULL UNIQUE,
password_hash VARCHAR(255) NOT NULL,
first_name VARCHAR(50),
last_name VARCHAR(50),
phone_number VARCHAR(20),
registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Plans (no foreign keys)
CREATE TABLE Plans (
plan_id INT AUTO_INCREMENT PRIMARY KEY,
plan_name VARCHAR(50) NOT NULL,
cpu_cores INT NOT NULL,
ram_gb INT NOT NULL,
storage_gb INT NOT NULL,
bandwidth_gb INT NOT NULL,
price_monthly DECIMAL(10, 2) NOT NULL
);

-- 3. UserPlans (references Users and Plans)
CREATE TABLE UserPlans (
user_plan_id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT NOT NULL,
plan_id INT NOT NULL,
start_date DATE NOT NULL,
expiry_date DATE NOT NULL,
auto_renew BOOLEAN DEFAULT FALSE,
FOREIGN KEY (user_id) REFERENCES Users(user_id),
FOREIGN KEY (plan_id) REFERENCES Plans(plan_id)
);

-- 4. Servers (no foreign keys)
CREATE TABLE Servers (
server_id INT AUTO_INCREMENT PRIMARY KEY,
server_name VARCHAR(50) NOT NULL,
ip_address VARCHAR(45) NOT NULL,
location VARCHAR(100),
status ENUM('active', 'maintenance', 'offline') DEFAULT 'active'
);

-- 5. UserServers (references Users, Servers, and UserPlans)
CREATE TABLE UserServers (
user_server_id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT NOT NULL,
server_id INT NOT NULL,
user_plan_id INT NOT NULL,
FOREIGN KEY (user_id) REFERENCES Users(user_id),
FOREIGN KEY (server_id) REFERENCES Servers(server_id),
FOREIGN KEY (user_plan_id) REFERENCES UserPlans(user_plan_id)
);

-- 6. Invoices (references Users and UserPlans)
CREATE TABLE Invoices (
invoice_id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT NOT NULL,
user_plan_id INT NOT NULL,
amount DECIMAL(10, 2) NOT NULL,
issue_date DATE NOT NULL,
due_date DATE NOT NULL,
status ENUM('paid', 'unpaid', 'overdue') DEFAULT 'unpaid',
FOREIGN KEY (user_id) REFERENCES Users(user_id),
FOREIGN KEY (user_plan_id) REFERENCES UserPlans(user_plan_id)
);

-- 7. PaymentMethods (references Users)
CREATE TABLE PaymentMethods (
payment_method_id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT NOT NULL,
method_type ENUM('credit_card', 'paypal', 'bank_transfer') NOT NULL,
details TEXT NOT NULL,
FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- 8. SupportTickets (references Users)
CREATE TABLE SupportTickets (
ticket_id INT AUTO_INCREMENT PRIMARY KEY,
user_id INT NOT NULL,
subject VARCHAR(100) NOT NULL,
description TEXT NOT NULL,
status ENUM('open', 'in_progress', 'closed') DEFAULT 'open',
created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- 9. ServerUsage (references UserServers)
CREATE TABLE ServerUsage (
usage_id INT AUTO_INCREMENT PRIMARY KEY,
user_server_id INT NOT NULL,
cpu_usage_percent DECIMAL(5, 2) NOT NULL,
ram_usage_percent DECIMAL(5, 2) NOT NULL,
storage_usage_gb DECIMAL(10, 2) NOT NULL,
bandwidth_usage_gb DECIMAL(10, 2) NOT NULL,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (user_server_id) REFERENCES UserServers(user_server_id)
);