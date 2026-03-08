import re
import requests
from bs4 import BeautifulSoup

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

def scrape_emails(url):
    try:
        headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache"
                    }

        response = requests.get(url, timeout=12, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        emails = set(re.findall(EMAIL_REGEX, text))

        if emails:
            return ", ".join(sorted(emails))
        else:
            return "No email found"

    except requests.exceptions.Timeout:
        return "Timeout – site not responding"

    except requests.exceptions.RequestException:
        return "Blocked or invalid URL"

    except Exception as e:
        return f"Error: {str(e)}"
