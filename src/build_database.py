"""Load cleaned rows into SQLite and run the portfolio SQL queries."""
import sqlite3
from .common import ROOT, DB_PATH, OUTPUT, load_clean_data


def main() -> None:
    data = load_clean_data()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script = (ROOT / "sql" / "churn_analysis.sql").read_text(encoding="utf-8")
    statements = [statement.strip() for statement in script.split(";") if statement.strip()]
    transcript = [f"Loaded {len(data):,} customers into SQLite.\n"]
    with sqlite3.connect(DB_PATH) as connection:
        data.to_sql("customers", connection, if_exists="replace", index=False)
        for i, query in enumerate(statements, start=1):
            # SQL comments above each statement describe the business question.
            cursor = connection.execute(query)
            headers = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            text = f"\nQuery {i}\n" + " | ".join(headers) + "\n"
            text += "\n".join(" | ".join(str(value) for value in row) for row in rows)
            transcript.append(text + "\n")
    summary = "\n".join(transcript)
    print(summary)
    (OUTPUT / "sql_results.txt").write_text(summary, encoding="utf-8")
    print(f"Database: {DB_PATH}\nQuery output: {OUTPUT / 'sql_results.txt'}")


if __name__ == "__main__":
    main()
