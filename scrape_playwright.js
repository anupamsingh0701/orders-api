const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  let totalSum = 0;

  for (let seed = 54; seed <= 63; seed++) {
    const url = `https://sanand0.github.io/tdsdata/js_table/?seed=${seed}`;
    console.log(`Visiting ${url}...`);
    await page.goto(url);
    await page.waitForSelector('table td');

    const cellTexts = await page.$$eval('td', elements => elements.map(el => el.textContent.trim()));
    let seedSum = 0;
    for (const text of cellTexts) {
      const num = parseFloat(text);
      if (!isNaN(num)) {
        seedSum += num;
      }
    }
    console.log(`Seed ${seed} sum: ${seedSum}`);
    totalSum += seedSum;
  }

  console.log(`Total sum of all tables across all seeds: ${totalSum}`);
  await browser.close();
})();
