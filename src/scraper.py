import time
import random
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
os.makedirs("data/raw", exist_ok=True)
os.makedirs("chrome_profile", exist_ok=True)

def init_driver():
    options = Options()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    current_path = os.getcwd()
    profile_path = os.path.join(current_path, "chrome_profile")
    options.add_argument(f"user-data-dir={profile_path}")
    options.add_argument("--start-maximized")
    options.add_argument("--lang=fr-FR")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def get_product_links(driver, keywords):
    all_links = set()
    base_url = "https://www.amazon.fr/s?k={}&page={}"
    max_pages_per_keyword = 5 
    
    print(f"--- Debut de la collecte ---")
    
    for keyword in keywords:
        print(f"\nTraitement : '{keyword}'")
        
        for page in range(1, max_pages_per_keyword + 1):
            url = base_url.format(keyword, page)
            print(f"Chargement page {page}...")
            
            try:
                driver.get(url)
                
                # Check simple anti-bot
                time.sleep(2)
                if "Robot" in driver.title:
                    print("Pause CAPTCHA (60s)...")
                    time.sleep(60)
                
                time.sleep(random.uniform(2, 4))
                
                if "Aucun résultat" in driver.page_source:
                    break

                product_cards = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
                
                for card in product_cards:
                    try:
                        links = card.find_elements(By.TAG_NAME, "a")
                        for link in links:
                            href = link.get_attribute("href")
                            if href and "/dp/" in href:
                                all_links.add(href)
                                break
                    except:
                        continue
                
                print(f"Liens trouves : {len(all_links)}")

            except Exception as e:
                print(f"Erreur : {e}")
                continue

    return list(all_links)

def scrape_single_product(driver, product_url):
    driver.get(product_url)
    time.sleep(random.uniform(1.5, 3))
    
    # Init dictionary with empty review
    data = {
        "url": product_url, 
        "title": "N/A", 
        "price": "N/A", 
        "rating": "N/A", 
        "review_count": "N/A",
        "first_review": "" # Added for NLP requirement
    }
    
    try:
        try: 
            data["title"] = driver.find_element(By.ID, "productTitle").text.strip()
        except: pass

        try:
            prices = driver.find_elements(By.CSS_SELECTOR, "span.a-price span.a-offscreen")
            if prices: 
                data["price"] = prices[0].get_attribute("textContent")
        except: pass

        try:
            rating_elm = driver.find_element(By.ID, "acrPopover")
            data["rating"] = rating_elm.get_attribute("title")
        except: pass
        
        try: 
            data["review_count"] = driver.find_element(By.ID, "acrCustomerReviewText").text
        except: pass

        # New block to grab the first review text
        try:
            review_elm = driver.find_element(By.CSS_SELECTOR, "div[data-hook='review-collapsed']")
            data["first_review"] = review_elm.text.strip()
        except: 
            pass

    except: pass
    return data

def main():
    keywords = [
        "clavier gamer",
        "clavier mécanique",
        "clavier gaming rgb",
        "clavier gamer logitech",
        "clavier gamer razer",
        "clavier gamer pas cher"
    ]
    
    try:
        driver = init_driver()
        
        # 1. Get links
        urls = get_product_links(driver, keywords)
        
        if not urls:
            print("Aucun lien trouve.")
            return

        print(f"\n--- Extraction details sur {len(urls)} produits ---")
        
        all_products = []
        
        for index, url in enumerate(urls):
            print(f"Produit {index + 1}/{len(urls)}")
            
            data = scrape_single_product(driver, url)
            all_products.append(data)
            
            # Save every 50 items
            if index % 50 == 0:
                pd.DataFrame(all_products).to_csv("data/raw/amazon_temp.csv", index=False)
        
        final_df = pd.DataFrame(all_products)
        final_df.to_csv("data/raw/amazon_raw_data.csv", index=False)
        print("Terminé ! Données sauvegardées.")
        
    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        try: driver.quit()
        except: pass

if __name__ == "__main__":
    main()