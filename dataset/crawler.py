"""
Python script for mass downloading .exes from download.cnet.com 
"""

import os
import requests
import html5lib
import time 
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.action_chains import ActionChains

"""
Also need headless firefox (geckodriver.exe) https://github.com/mozilla/geckodriver/releases
 and adblock xpi  (adblock_plus-3.10.2-an+fx.xpi) https://adblockplus.org/en/download 
"""

"""
main_download_url = "https://download.cnet.com/windows/?sort=mostPopular&price=free"
# Only want windows, sorted by popularity and free software

list_of_apps = []

response = requests.get(main_download_url)

soup = BeautifulSoup(response.content, 'html5lib')
#print(soup.prettify())

table = soup.find('div', class_='g-grid-container u-grid-columns c-searchResults')




# Grab app names and download urls

for row in table.findAll('div', class_='c-globalCard lg:u-col-3 md:u-col-3 sm:u-col-2 c-productCard u-flexbox-column c-productCard-detailed'):
    app_name = row.find('div', class_='c-productCard_info')
    print(app_name.h3.string)
    app_page_link = row.find('a', class_='c-productCard_link')
    print(app_page_link['href'])
"""

# Instantiate a Firefox options object so you can set the size and headless preference
# change the <path_to_download_default_directory> to whatever your default download folder is located
options = FirefoxOptions()

options.add_argument("--headless")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-notifications")
options.add_argument('--no-sandbox')
options.add_argument('--verbose')

options.add_argument('--disable-gpu')
options.add_argument('--disable-software-rasterizer')

download_folder = "C:\\Users\\Emily\\Documents\\UCL\\ncc-project\\cdt-nccgroup\\dataset\\cleanware"
gecko_path = r'C:\Users\Emily\Documents\UCL\ncc-project\cdt-nccgroup\dataset\geckodriver.exe'
adblockfile = "C:\\Users\\Emily\\Documents\\UCL\\ncc-project\\cdt-nccgroup\\dataset\\adblock_plus-3.10.2-an+fx.xpi"

profile = webdriver.FirefoxProfile()
profile.set_preference('browser.download.folderList', 2) # custom location
profile.set_preference('browser.download.manager.showWhenStarting', False)
profile.set_preference('browser.download.dir', download_folder)
profile.set_preference('browser.helperApps.alwaysAsk.force', False)
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/x-msdownload')
profile.add_extension(extension=adblockfile)
profile.set_preference("extensions.adblockplus.currentVersion", "3.1")


# Initialize driver object and change the <path_to_gecko_driver> depending on location of gecko driver
driver = webdriver.Firefox(options=options, executable_path=gecko_path, firefox_profile=profile)
print("Headless Firefox Initialised")

driver.install_addon(adblockfile)
# Note that adblock opens a new tab 


driver.get("https://download.cnet.com/CCleaner/3000-18512_4-10315544.html")
print("Loading website: " + driver.title)

timeout = 30

# Check page has loaded
try:
    element_present = EC.presence_of_element_located((By.CSS_SELECTOR,'#onetrust-accept-btn-handler'))
    WebDriverWait(driver, timeout).until(element_present)
    print("Could find the button and loaded properly")
except NoSuchElementException:
    print('Could not find that element')
except TimeoutException:
    print("Timed out waiting for page to load")


# Going to have to click the cookies button

cookie_accept = driver.find_element_by_css_selector('#onetrust-accept-btn-handler')
driver.execute_script("arguments[0].scrollIntoView();", cookie_accept)

js_click = "arguments[0].click();"
driver.execute_script(js_click,cookie_accept)

# Put in these in try/except to catch long wait times

download_button = driver.find_element_by_css_selector('.c-globalButton.g-text-small.c-productActionButton_button.c-productActionButton_button-standard.u-text-uppercase.c-globalButton-medium.c-globalButton-standard')
driver.execute_script("arguments[0].scrollIntoView();", download_button)
driver.execute_script(js_click,download_button)

# remember to rejig environment yml 