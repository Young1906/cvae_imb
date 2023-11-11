import requests
import urllib

# Telegram TOKEN
TOKEN = "6354404240:AAF4rtZsOKuFX4jVdd_qoP9k9g6M71AL6fI"
GROUP_ID = "-1001751779191"
URL = (
        "https://api.telegram.org/bot{token}"
        .format(token=TOKEN)
    )


def send_message(message: str) -> int:
    # escape
    message = urllib.parse.quote(
            message.encode('utf-8'))

    url = (
            "{base}/sendMessage?chat_id={group_id}&text={msg}"
            .format(
                base=URL,
                group_id=GROUP_ID,
                msg=message)
        )

    res = requests.get(url)
    return res.status_code



if __name__ == "__main__":
    send_message("Test")

