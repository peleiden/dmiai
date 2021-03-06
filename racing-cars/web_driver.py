import os
import shutil
import sys
import time

from selenium import webdriver
import warnings
warnings.filterwarnings("ignore")

import logging

SERVICE = "https://api.dmiai.dk/racing-game"
HOST = "127.0.0.1"

def start_driver(port: int, seed: int=0):
    print("Starting Selenium server ...")
    driver = webdriver.Firefox(os.path.split(shutil.which("firefox"))[0])
    print(driver)
    driver.get(SERVICE)
    print("driver")
    # Fill in data
    for _id, data in zip(("random-seed", "api-protocol", "api-host", "api-port"), (seed, "http", HOST, port)):
        print(_id)
        elem = driver.find_element_by_id(_id + "-input-field")
        elem.send_keys(data)
    print("loop")
    # Enable loop
    driver.find_element_by_id("auto-restart-button").click()
    driver.refresh()
    print("heals")
    # Perform health check
    driver.find_element_by_id("health-check-button").click()
    time.sleep(2)
    print("status")
    status = driver.find_element_by_id("status-message").get_attribute("innerHTML")
    assert "Ready" in status, f"Health check failed!, status was {status}"
    # Start the server
    driver.find_element_by_id("start-button").click()
    print("Selenium driver running!")

if __name__ == "__main__":
    time.sleep(5)
    os.environ['MOZ_HEADLESS'] = '1'
    port, seed = sys.argv[1:]
    start_driver(port, seed)
