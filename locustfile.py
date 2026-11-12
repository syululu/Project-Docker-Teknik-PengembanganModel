
from locust import HttpUser, task, between


class CardioCheckUser(HttpUser):
    """
    Simulasi pengguna yang mengakses CardioCheck
    melalui Nginx Gateway.
    """

    # Jeda antar-task setiap virtual user
    wait_time = between(1, 3)

    @task(5)
    def access_homepage(self):
        """
        Menguji akses halaman utama Streamlit.
        """
        with self.client.get(
            "/",
            name="Homepage - CardioCheck",
            catch_response=True
        ) as response:

            if response.status_code == 200:
                if "text/html" in response.headers.get(
                    "Content-Type", ""
                ).lower():
                    response.success()
                else:
                    response.failure(
                        "Response bukan HTML"
                    )
            else:
                response.failure(
                    f"Homepage error: HTTP {response.status_code}"
                )

    @task(1)
    def check_health(self):
        """
        Menguji kondisi kesehatan aplikasi Streamlit.
        """
        with self.client.get(
            "/_stcore/health",
            name="Health Check - Streamlit",
            catch_response=True
        ) as response:

            if response.status_code == 200:
                if response.text.strip().lower() == "ok":
                    response.success()
                else:
                    response.failure(
                        "Health check tidak mengembalikan OK"
                    )
            else:
                response.failure(
                    f"Health check error: HTTP {response.status_code}"
                )
