from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep

options = Options()
options.add_argument("start-maximized")

driver = webdriver.Chrome(options=options)

try:
    url = "https://dir.indiamart.com/search.mp?ss=Bakery+Oven"
    driver.get(url)

    input("\nComplete IndiaMART login in Chrome, then press Enter here...")

    print("\nPAGE TITLE:", driver.title)
    print("\nCURRENT URL:", driver.current_url)

    print("\nPAGE TEXT:")
    print(driver.find_element(By.TAG_NAME, "body").text[:3000])

    print("\nPRODUCT CARD COUNT:")
    print(driver.execute_script(
        "return document.querySelectorAll('.template7-product-card').length"
    ))
    print("\nMLP PRODUCT CARD COUNT:")
    mlp_cards = driver.find_elements(
        By.CSS_SELECTOR, "article.mlpCard"
        )
    print(len(mlp_cards))
    print("\nFIRST PRODUCT CARD HTML:\n")
    if mlp_cards:
        print(mlp_cards[0].get_attribute("outerHTML")[:12000])
    else:
        print("No MLP product cards found.")
    cards = driver.find_elements(
        By.CSS_SELECTOR, "div.template7-product-card"
        )
    if cards:
        print(cards[0].get_attribute("outerHTML")[:12000])
    else:
        print("No template7-product-card elements found.")
    print("\nPRODUCT-RELATED HTML ELEMENTS:")
    elements = driver.execute_script("""
        return Array.from(document.querySelectorAll('[class]'))
            .filter(el =>
                /product|card|listing|item/i.test(String(el.className)) &&
                el.innerText.trim().length > 0
            )
            .slice(0, 40)
            .map(el => ({
                tag: el.tagName,
                className: String(el.className),
                text: el.innerText.substring(0, 180)
            }));
    """)

    for element in elements:
        print(element)

    print("\nHTML SAMPLE:")
    print(
        driver.find_element(By.TAG_NAME, "body")
        .get_attribute("innerHTML")[:5000]
    )

    input("\nPress Enter to close the browser...")

finally:
    driver.quit()