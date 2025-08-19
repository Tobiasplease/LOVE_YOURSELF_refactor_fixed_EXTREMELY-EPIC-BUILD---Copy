Nu börjar jag få till det med home, nollpunkt och origin.

Man sätter nollpunkt och så i arduinon (GRBL heter programvaran) därför blir det nog svårt att testa utan roboten.

det verkar som att det funkar och ligger kvar när man startar bCNC, dock är jag inte helt säker på att det kommer att funka med origin offset,

så eventuellt får vi sätta det i g-koden så det finns med när vi laddar in filen. att HOME-a måste man nog göra som ett seriellt kommando medans nollpunkt och origin osv är g-kod som då borde gå att baka in t ex alla våra filer.

Arduino-programmet heter GRBL robottini som är en fork på GRBL för att kunna hantera servot som lyfter pennan. Det är själva CNC-maskinen. Arduino Uno

---

Nu funkar det ganska bra med kod och roboten men det behöver säker pillas lite till när vi har ett filformat som AI:n spottar ut. Mitt python-skript använder sig bara av CNC-programmet och konverterar filen i den innan den byter lite kommandon med hjälp av samma py-skript, så det ska passa vår ”hand” som håller i pennan. Och sen laddar den in filen igen och kör den

Jag fick inte till det (eller chat gpt) att fixa det med python/api, utan fick kontrollera programmet i en kommand-prompt i GUIn. Lite muppigt och just nu så är den lite beroende av olika pauser som kan behöva kalibreras.

Så jag kontrollerar den med pyautogui

---

Här kommer koden som jag testkör nu. bCNC körs automatiskt igång när datorn startar men jag behöver antagligen också skriva ett enkelt program som HOME-ar maskinen. Det borde inte behövas köras varje gång men vi kan ju se.

Koden laddar in ORIGINAL.SVG från input-mappen och sparar den till RAW.NGC så att jag får en ok G-KOD.

Sen letar python-programmet igenom filen för att göra om Z-axelrörelsen till att istället lyfta och sänka pennan och sparar den nya koden DRAWING.NGC. Här behöver jag fixa till lite till, hastighet och lite grejer behövas sättas i början av filen men det är enkelt, vi behöver kolla lite på hur snabbt maskinen behöver köra (den tappar lite precision om man kör den för snabbt)

Sen laddas DRAWING.NGC i bCNC och körs.

Det gick ju inte att få till det i API så jag kör allt i GUI med framförallt kommandoraden i programmet med hjälp av PYAUTOGUI.

Här är en lista på vad man kan göra i kommandoraden:

https://github-wiki-see.page/m/vlachoudis/bCNC/wiki/CommandLine

---
