from flask import Flask, render_template, request, send_file, flash
import pandas as pd
import os
from scraper.main import scrape_emails
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "mvp-secret-key"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    single_result = None

    if request.method == "POST":

        # 🔹 Single URL Mode
        if "single_url" in request.form:
            url = request.form.get("single_url")
            single_result = scrape_emails(url)

        # 🔹 Excel Mode
        if "file" in request.files:
            file = request.files["file"]

            if file.filename.endswith(".xlsx"):

                # sanitize filename
                filename = secure_filename(file.filename)
                if filename == "":
                    flash("No file selected")
                    return render_template("index.html")

                # save upload
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)

                # read safely
                try:
                    df = pd.read_excel(path)
                except Exception as e:
                    flash(f"Failed to read Excel file: {str(e)}")
                    os.remove(path)
                    return render_template("index.html")

                # delete uploaded file immediately after reading
                

                # check required column
                if "urls" not in df.columns:
                    flash("Excel must contain 'urls' column")
                    return render_template("index.html")

                # scrape emails
                emails = []
                for url in df["urls"]:
                    emails.append(scrape_emails(str(url)))
                    time.sleep(2)  # prevent blocking

                df["emails"] = emails

                # save unique output file
                output_filename = f"email_results_{int(time.time())}.xlsx"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                df.to_excel(output_path, index=False)

                return send_file(output_path, as_attachment=True)

            else:
                flash("Only .xlsx files are allowed")

    return render_template(
        "index.html",
        single_result=single_result
    )

if __name__ == "__main__":
    app.run(debug=False)
