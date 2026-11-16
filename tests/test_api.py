#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for Magic Data Assistant API.
Supports both offline unit tests and live integration tests.
Usage:
    python -m unittest tests/test_api.py       # Run unit tests
    python tests/test_api.py --live             # Run live integration tests against running server
    python tests/test_api.py --deep             # Run deep multi-sheet test
"""

import os
import sys
import json
import time
import random
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# Base URL for integration tests
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
UPLOAD_ENDPOINT = f"{BASE_URL}/upload-file/"
CHAT_ENDPOINT = f"{BASE_URL}/chat/"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
TABLES_ENDPOINT = f"{BASE_URL}/tables/"


# --- ANSI Colors ---
class bcolors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    WARNING = '\033[93m'
    INFO = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}===== {text} ====={bcolors.ENDC}")


def print_pass(text):
    print(f"{bcolors.OKGREEN}[PASS]{bcolors.ENDC} {text}")


def print_fail(text, response_text=None):
    print(f"{bcolors.FAIL}[FAIL]{bcolors.ENDC} {text}")
    if response_text:
        print(f"       {bcolors.WARNING}Response: {str(response_text)[:300]}...{bcolors.ENDC}")


def print_info(text):
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} {text}")


# =====================================================================
# Unit Tests (Self-contained, no running server required)
# =====================================================================
class UnitTests(unittest.TestCase):
    """Offline unit tests for data cleaning, sanitization, and safety."""

    def test_clean_currency_detection(self):
        """Verify currency is cleaned while preserving normal text with commas."""
        from main import clean_dataframe

        df = pd.DataFrame({
            'city': ['San Francisco, CA', 'New York, NY', 'Tokyo, Japan'],
            'salary': ['$90,000', '$75,000.50', None],
            'rupee_val': ['₹1,500.00', '₹2,500.00', '₹3,000'],
            'plain_num': ['100', '200', '300'],
            'notes': ['Item, with comma', 'Another, item', 'Final, note']
        })

        cleaned = clean_dataframe(df)

        # Cities and notes with commas MUST NOT be converted to NaN
        self.assertEqual(cleaned['city'].iloc[0], 'San Francisco, CA')
        self.assertEqual(cleaned['notes'].iloc[0], 'Item, with comma')

        # Currency columns must be converted to float
        self.assertTrue(pd.api.types.is_float_dtype(cleaned['salary']))
        self.assertEqual(cleaned['salary'].iloc[0], 90000.0)
        self.assertEqual(cleaned['salary'].iloc[1], 75000.5)

        self.assertTrue(pd.api.types.is_float_dtype(cleaned['rupee_val']))
        self.assertEqual(cleaned['rupee_val'].iloc[0], 1500.0)

    def test_sql_safety_check(self):
        """Verify that destructive SQL operations are rejected."""
        from main import is_query_safe

        safe_queries = [
            'SELECT * FROM "sales" LIMIT 10;',
            'SELECT customer_id, SUM(amount) FROM "orders" GROUP BY customer_id;',
            'WITH top_emp AS (SELECT * FROM "employees") SELECT * FROM top_emp;'
        ]
        for q in safe_queries:
            is_safe, _ = is_query_safe(q)
            self.assertTrue(is_safe, f"Safe query was flagged as unsafe: {q}")

        unsafe_queries = [
            'DROP TABLE "sales";',
            'DELETE FROM "users" WHERE id = 1;',
            'TRUNCATE "orders";',
            'ALTER TABLE "employees" DROP COLUMN salary;',
            'UPDATE "users" SET role = "admin";'
        ]
        for q in unsafe_queries:
            is_safe, forbidden = is_query_safe(q)
            self.assertFalse(is_safe, f"Unsafe query was permitted: {q}")
            self.assertTrue(len(forbidden) > 0)

    def test_pandas_to_sql_type(self):
        """Verify pandas to PostgreSQL type mapping."""
        from main import pandas_to_sql_type

        self.assertEqual(pandas_to_sql_type(pd.Series([1, 2, 3]).dtype), "BIGINT")
        self.assertEqual(pandas_to_sql_type(pd.Series([1.5, 2.5]).dtype), "DOUBLE PRECISION")
        self.assertEqual(pandas_to_sql_type(pd.Series([True, False]).dtype), "BOOLEAN")
        self.assertEqual(pandas_to_sql_type(pd.to_datetime(pd.Series(['2026-01-01'])).dtype), "TIMESTAMP")
        self.assertEqual(pandas_to_sql_type(pd.Series(['hello', 'world']).dtype), "TEXT")


# =====================================================================
# Live Integration Test Suite (Requires running server)
# =====================================================================
class LiveTestSuite:
    """End-to-end integration tests against a live server."""

    def __init__(self):
        self.test_files = []
        try:
            from faker import Faker
            self.fake = Faker()
        except ImportError:
            self.fake = None

    def setUp(self):
        print_header("Setting up Live Test Environment")
        # Create employees.csv
        with open("employees.csv", "w", encoding="utf-8") as f:
            f.write("employee_id,employee_name,department\n")
            f.write("1,Anbu,Engineering\n")
            f.write("2,Bala,Sales\n")
        self.test_files.append("employees.csv")

        # Create salaries.xlsx
        salaries_data = {'employee_id': [1, 2], 'salary_inr': [90000, 75000]}
        df_salaries = pd.DataFrame(salaries_data)
        with pd.ExcelWriter('salaries.xlsx', engine='openpyxl') as writer:
            df_salaries.to_excel(writer, sheet_name='salary_data', index=False)
        self.test_files.append("salaries.xlsx")

        # Create lifecycle_test.csv
        with open("lifecycle_test.csv", "w", encoding="utf-8") as f:
            f.write("version\n1\n")
        self.test_files.append("lifecycle_test.csv")

        print_pass("Test files created.")

    def tearDown(self):
        print_info("\nTearing down Live Test Environment")
        for f in self.test_files:
            if os.path.exists(f):
                os.remove(f)
        for i in range(5):
            fname = f"user_{i+1}_data.csv"
            if os.path.exists(fname):
                os.remove(fname)
        print_pass("Cleanup complete.")

    def _upload_file(self, filepath, mime_type):
        with open(filepath, 'rb') as f:
            return requests.post(UPLOAD_ENDPOINT, files={'file': (os.path.basename(filepath), f, mime_type)}, timeout=60)

    def _run_chat_test(self, test_name, payload, success_condition, retries=3, delay=5):
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
                if response.status_code != 200:
                    print_fail(f"{test_name} - Request failed with status {response.status_code}.", response.text)
                    continue

                try:
                    response_data = response.json()
                    answer = response_data.get("answer", "")
                    if success_condition(answer):
                        print_pass(f"{test_name} successful.")
                        print_info(f"Response: {answer[:100]}...")
                        return True
                    else:
                        print_fail(f"{test_name} failed. Final answer did not meet condition.", answer)
                except json.JSONDecodeError:
                    print_fail(f"{test_name} failed to decode JSON.", response.text)
            except Exception as e:
                print_fail(f"An exception occurred during '{test_name}' test: {e}")
            if attempt < retries:
                print_info(f"Retrying '{test_name}' in {delay} seconds (attempt {attempt + 1} of {retries})...")
                time.sleep(delay)
        return False

    def test_health(self):
        print_header("Test 0: Health Check Endpoint")
        try:
            res = requests.get(HEALTH_ENDPOINT, timeout=10)
            if res.status_code == 200:
                print_pass(f"Health check OK: {res.json()}")
            else:
                print_fail(f"Health check failed with {res.status_code}: {res.text}")
        except Exception as e:
            print_fail(f"Health check error: {e}")

    def test_core_functionality(self):
        print_header("Test 1: Core Functionality (Upload & Simple Query)")
        res_csv = self._upload_file("employees.csv", "text/csv")
        assert res_csv.status_code == 200, f"CSV Upload failed: {res_csv.text}"
        print_pass("CSV upload successful.")

        res_xlsx = self._upload_file("salaries.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        assert res_xlsx.status_code == 200, f"XLSX Upload failed: {res_xlsx.text}"
        print_pass("XLSX upload successful.")

        self._run_chat_test(
            "Simple chat query",
            {"table_names": ["employees"], "question": "Who is in the Sales department?"},
            lambda answer: "bala" in answer.lower()
        )

    def test_table_lifecycle(self):
        print_header("Test 2: Table Lifecycle (Overwrite Test)")
        self._upload_file("lifecycle_test.csv", "text/csv")

        self._run_chat_test(
            "Version 1 Query",
            {"table_names": ["lifecycle_test"], "question": "What is the value in the version column?"},
            lambda answer: "1" in answer
        )

        with open("lifecycle_test.csv", "w", encoding="utf-8") as f:
            f.write("version\n2\n")
        self._upload_file("lifecycle_test.csv", "text/csv")

        self._run_chat_test(
            "Version 2 Query",
            {"table_names": ["lifecycle_test"], "question": "What is the value in the version column?"},
            lambda answer: "2" in answer
        )


if __name__ == "__main__":
    if "--live" in sys.argv:
        suite = LiveTestSuite()
        try:
            suite.setUp()
            suite.test_health()
            suite.test_core_functionality()
            suite.test_table_lifecycle()
        finally:
            suite.tearDown()
    else:
        unittest.main(argv=[sys.argv[0]])
