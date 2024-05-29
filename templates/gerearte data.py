import pandas as pd
import random

# Provided names
names = ["Max", "Sophie", "Ezra", "Ivy", "Caleb", "Luna", "Theo", "Nova", "Asher", "Aria",
"Lincoln", "Lola", "Leo", "Aurora", "Eli", "Jade", "Finn", "Willow", "Axel", "Hazel",
"Hudson", "Sage", "Micah", "Isla", "Ryder", "Elise", "Sawyer", "Gemma", "Roman", "Daisy",
"Kai", "Iris", "Beckett", "Violet", "Jaxon", "Stella", "Xander", "Cora", "Declan", "Wren",
"Silas", "Freya", "Milo", "Maeve", "Zane", "Evangeline", "Archer", "Poppy", "Kingston", "Juniper"]

# Generate 50 records
records = []
for _ in range(50):
    name = random.choice(names)
    branch = "AIML"
    cgpa = round(random.uniform(6.0, 10.0), 2)
    leet_rank = random.randint(1, 100)
    dsa_score = random.randint(1, 100)
    cns_score = random.randint(1, 100)
    dbms_score = random.randint(1, 100)
    internship = random.choice(["CLOUD", "DEVOPS", "WEB_DEVELOPMENT", "CYBER_SECURITY"])
    
    records.append([name, branch, cgpa, leet_rank, dsa_score, cns_score, dbms_score, internship])

# Create DataFrame
df = pd.DataFrame(records, columns=["NAME", "BRANCH", "CGPA", "LEET_RANK", "DSA", "CNS", "DBMS", "INTERNSHIP"])

# Write to Excel
with pd.ExcelWriter('student_records.xlsx', engine='openpyxl') as excel_writer:
    df.to_excel(excel_writer, index=False, sheet_name='Sheet1')

print("Excel file 'student_records.xlsx' created successfully!")
