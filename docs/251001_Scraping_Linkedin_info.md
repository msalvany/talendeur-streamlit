# Research: Scraping LinkedIn Profiles

_Date: 2025-10-01_   

## 📝 Task / Focus
Gather info about sacraping CV data from Linkedin personal profiles

## Why scrape Linkedin?

LinkedIn holds structured professional data: names, job titles, companies, education, skills.
For HR, recruitment, and data analysis, automating the extraction can save huge amounts of time.

⚠️ Scraping LinkedIn directly raises legal and ethical issues → it’s against their Terms of Service.
That means: we need to carefully consider how we access and use this data.

## Approaches
**A. Official API (LinkedIn Developer API)**
- Only for approved apps, requires OAuth authentication.
- Data access is very limited (basic profile fields, mostly for integrations like “Login with LinkedIn”).
- Useful if the project doesn’t need much beyond basic user info.

**B. Unofficial APIs / Libraries**

- What they are:

    Open-source Python (or other language) packages that reverse-engineer LinkedIn’s internal APIs or simulate requests.
Example: [linkedin-api](https://pypi.org/project/linkedin-api/) 
- How they work:
    1. Pretend to be a browser or LinkedIn mobile app.
    2. Send HTTP requests to LinkedIn’s hidden endpoints.<>
    3. Return data in JSON (structured, clean).
- **Pros**
    - Code runs locally, full control.
    - You can customize exactly what data you pull.
    - No external dependency (beyond LinkedIn itself).
- **Cons**
    - Unofficial → violates LinkedIn’s ToS.
    - Very fragile: breaks whenever LinkedIn changes its private API.
    - Risk of account blocks or CAPTCHAs.

- ⚠️ Risk: LinkedIn can block accounts if usage is detected.

**C. Automation Tools / Scraping Services**

- What they are:

    Ready-made tools or paid services that handle the scraping for you.

- How they work:
    - Some run headless browsers (Selenium-like) on the cloud.
    - Some have pre-built “actors”/APIs for LinkedIn.
    - You configure them with parameters (e.g., list of profile URLs), and they return structured data.

- **Pros**: reliable, maintained, no need to fight HTML changes.
- **Cons**: cost, third-party dependency.

- Examples:

    - [Phantombuster](https://phantombuster.com/phantombuster?category=linkedin) → cloud-based automation, ready-made recipes for LinkedIn profile scraping.

        "Find, filter and connect with people engaging on LinkedIn - ready to launch in seconds, with full control over every step."
    - [Apify](https://docs.apify.com/) → scrapers for LinkedIn jobs, companies, and profiles.

        "Apify is the largest ecosystem where developers build, deploy, and publish web
scrapers, AI agents, and automation tools. We call them Actors."

**D. Web Scraping with Selenium / Playwright**

- What they are:
    - Open-source tools that let you control a real browser (Chrome, Firefox, Edge) programmatically.
    - You write code to “open LinkedIn → log in → click → scroll → copy the HTML.”
    - They simulate a human user browsing the site, but without you touching the mouse.

- How they work:
    - **[Selenium](https://www.selenium.dev/documentation/)**: older, very widely used. Works with many languages (Python, Java, JS).
    - **[Playwright](https://playwright.dev/docs/intro)**: newer (by Microsoft), faster, better support for modern web apps, auto-waits for elements.

    Both:
    - Run a headless browser (optional → no UI).
    - Extract the HTML/DOM, which you can then parse with something like BeautifulSoup.


- **Pros:**
    - Full control → you can scrape anything visible on the page.
    - Can handle dynamic content (infinite scroll, JavaScript rendering).
    - Good for prototyping “what’s possible.”

- **Cons:**
    - Very fragile (LinkedIn changes selectors often).
    - ⚠️ Easy to get detected → LinkedIn may block or shadow-ban accounts.
    - Slower than APIs (loading full pages, images, JS).
    - You have to manage proxies, delays, logins, captchas yourself.

**Summary**
- **Unofficial API libraries** = trick LinkedIn’s backend → JSON output.
- **Selenium / Playwright** = trick LinkedIn’s frontend → HTML scraping.
- Services like **Phantombuster** = they run Selenium/Playwright-like logic in the cloud for you, packaged nicely.

## Possible Pipeline for Safe Use
1.	Check if user provides LinkedIn data (e.g., they upload their CV or share a profile URL).
2.	For research/demo purposes:
    - Use linkedin-api to fetch a few test fields.
    - Or test a service like Proxycurl’s free tier.
3.	For production / legal compliance:
    - Prefer user-uploaded data (PDF CVs).
    - Or apply to LinkedIn’s official API (though access is restrictive).

## Testing strategy I would use
1. Detect the **PDF type** (text vs scanned).
    - Try reading with pdfplumber.
    - If result is empty → assume scanned, apply OCR.
2. **Normalize text** (remove line breaks, fix encoding issues).
3. Apply **regex/NLP** to extract structured info:
    - CVs: name, email, phone, skills.
    - Invoices: vendor, total, date.

## Takeaways
- Unlike PDFs, scraping LinkedIn is less of a technical challenge and more of a legal/ethical challenge.
- For an internship context, safest approach is to:
    - Focus on PDF scraping (user-provided data).
    - Explore LinkedIn scraping theoretically or via third-party APIs (so you learn how, but avoid breaking TOS).
- If the project really needs LinkedIn profile info, using a service like Proxycurl is more realistic than building a scraper from scratch.