"""
Python script for mass downloading .exes from download.cnet.com 
Probably lots of malware...
"""

import os
import requests
import html5lib
import time 
import itertools # temp to slice my dict
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

download_folder = "C:\\Users\\Emily\\Documents\\UCL\\ncc-project\\cdt-nccgroup\\dataset\\downloads"
gecko_path = "C:\\Users\\Emily\\Documents\\UCL\\ncc-project\\cdt-nccgroup\\dataset\\geckodriver.exe"
adblock_path = "C:\\Users\\Emily\\Documents\\UCL\\ncc-project\\cdt-nccgroup\\dataset\\adblock_plus-3.10.2-an+fx.xpi"

# Change to point to default download dir, gecko and adblock files. I think paths need to be absolute. 
#download_folder = <path_to_download_default_directory> 
#gecko_path = <path_to_gecko_sdriver>
#adblockfile = <path_to_adblock_addon>

def scrape_app_urls(website_url):
    app_list = {}

    response = requests.get(website_url)

    soup = BeautifulSoup(response.content, 'html5lib')
    #print(soup.prettify())

    table = soup.find('div', class_='g-grid-container u-grid-columns c-searchResults')

    # Grab app names and download urls
    for row in table.findAll('div', class_='c-globalCard lg:u-col-3 md:u-col-3 sm:u-col-2 c-productCard u-flexbox-column c-productCard-detailed'):
        app_name = row.find('div', class_='c-productCard_info')
        app_page_link = row.find('a', class_='c-productCard_link')
        app_list[str(app_name.h3.string)] = str(app_page_link['href'])

    return(app_list)

def driver_setup(download_folder,gecko_path,adblock_path):
    # Instantiate a Firefox options object so you can set the size and headless preference
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-notifications")
    options.add_argument('--no-sandbox')
    options.add_argument('--verbose')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')

    accepted_mime_types_for_download = "application/xml,application/octet-stream,application/x-msdownload"
    
    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2) # custom location
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir', download_folder)
    profile.set_preference('browser.helperApps.alwaysAsk.force', False)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', accepted_mime_types_for_download)
    profile.add_extension(extension=adblock_path)
    profile.set_preference("extensions.adblockplus.currentVersion", "3.1")

    # Initialize driver object and change exe path to <path_to_gecko_driver> 
    driver = webdriver.Firefox(options=options, executable_path=gecko_path, firefox_profile=profile)

    driver.install_addon(adblock_path)
    # Note that adblock opens a new tab 
    print("Headless Firefox Initialised")

    return(driver)

# remove this hardcoding
def download_wait(timeout, nfiles, directory):
    """
    Wait for downloads to finish with a specified timeout.

    Args
    ----
    directory : str
        The path to the folder where the files will be downloaded.
    timeout : int
        How many seconds to wait until timing out.
    nfiles : int, defaults to None
        If provided, also wait for the expected number of files.

    """
    print("Waiting for file to download")
    print("=========================")
    seconds = 0
    dl_wait = True
    while dl_wait and seconds < timeout:
        time.sleep(1)
        dl_wait = False
        files = os.listdir(directory)
        if nfiles and len(files) != nfiles:
            dl_wait = True

        for fname in files:
            if fname.endswith('.exe.part'):
                dl_wait = True
        seconds += 1
    return seconds


def file_download(driver, app_url,num_files,download_folder):
    domain_url = "https://download.cnet.com"
    app_url = str(domain_url) + str(app_url)
    
    driver.get(app_url)
    print("Loading website: " + driver.title)

    timeout = 10

    # Check page has loaded and click cookie button, sometimes can skip cookie
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR,'#onetrust-accept-btn-handler'))
        WebDriverWait(driver, timeout).until(element_present)
        print("Could find the cookie button, page loaded properly")
        print("=========================")
    except TimeoutException:
        print("Timed out waiting for page to load, or no cookie banner")
        pass
    else:
        cookie_accept = driver.find_element_by_css_selector('#onetrust-accept-btn-handler')
        driver.execute_script("arguments[0].scrollIntoView();", cookie_accept)
        js_click = "arguments[0].click();"
        driver.execute_script(js_click,cookie_accept)

    # Put in below in try/except to catch long wait times?

    # Check for download button
    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR,'.c-globalButton.g-text-small.c-productActionButton_button.c-productActionButton_button-standard.u-text-uppercase.c-globalButton-medium.c-globalButton-standard'))
        WebDriverWait(driver, timeout).until(element_present)
        print("Found download button")
        print("=========================")
    except TimeoutException:
        print("Timed out looking for download button")
        print("=========================")

    js_click = "arguments[0].click();"
    download_button = driver.find_element_by_css_selector('.c-globalButton.g-text-small.c-productActionButton_button.c-productActionButton_button-standard.u-text-uppercase.c-globalButton-medium.c-globalButton-standard')
    driver.execute_script("arguments[0].scrollIntoView();", download_button)
    driver.execute_script(js_click,download_button)

    # give some time to complete download
    download_wait(20,num_files,download_folder)
    

main_download_url = "https://download.cnet.com/windows/2/?sort=mostPopular&price=free"
# Only want windows, sorted by popularity and free software
# Change this to scrape next set of apps
# p1 https://download.cnet.com/windows/?sort=mostPopular&price=free
# p2 https://download.cnet.com/windows/2/?sort=mostPopular&price=free
# p3 https://download.cnet.com/windows/3/?sort=mostPopular&price=free
# etc... 


app_list = scrape_app_urls(main_download_url)
#print(app_list)

#app_list = dict(itertools.islice(app_list.items(), 5))

driver = driver_setup(download_folder,gecko_path,adblock_path)


#Check what number already present

num_files = len(os.listdir(download_folder))+1
#print('number of files expected',num_files)

for app, app_url in app_list.items():
    # app_urls look like /CCleaner/3000-18512_4-10315544.html
    #print("URL",app_url)
    print("Downloading ",app)
    file_download(driver, app_url, num_files,download_folder)
    num_files=len(os.listdir(download_folder))+1


driver.close()