import wikipedia
import time

wikipedia.set_lang("en")

#The list( page_titles ) given below contains the topic we need to scrape from wikipedia.
#These topics are not manually identified i have used an llm to identify and extract topics based on
#prompts given in test and train data.


page_titles = [
    # Astrophysics & Cosmology
    "James Webb Space Telescope", "Modified Newtonian dynamics", 
    "Gravity Probe B", "Lunar Laser Ranging experiment", "Deep Space Network", 
    "Milky Way", "Roche limit", "Crab Pulsar", "Schwarzschild radius", 
    "Kapteyn universe", "Redshift", "Doppler effect",
    
    # Quantum Mechanics & Particle Physics
    "Higgs boson", "Standard Model", "Uncertainty principle", 
    "Wigner's theorem", "Hilbert space", "Hamiltonian (quantum mechanics)", 
    "CP violation", "Yoichiro Nambu", "Josephson effect", 
    "Supersymmetric quantum mechanics", "Larmor precession",
    
    # Classical Mechanics, Relativity & Fluid Dynamics
    "Navier–Stokes equations", "Lorentz transformation", "Minkowski space", 
    "Special relativity", "General relativity", "Kutta condition", 
    "Hooke's law", "Liouville's theorem (Hamiltonian)", "Fermat's principle",
    
    # Thermodynamics & Materials Science
    "Stefan–Boltzmann law", "Maxwell's demon", "Leidenfrost effect", 
    "Fischer–Tropsch process", "Carnot heat engine", "Memristor", 
    "Landau–Lifshitz–Gilbert equation", "Optical signal-to-noise ratio",
    
    # Miscellaneous / Cross-Disciplinary
    "Coordinated Universal Time", "Martin Heidegger", "Interleukin 10"
]



for title in page_titles:
    try:
        page=wikipedia.page(title,auto_suggest=True) #extracts the page content
        with open(f"{title}.txt","w",encoding="utf-8")as f:
            f.write(page.content) #opening and wriring the extarcted content to a file with topic as title
        print(f"successfully scraped and saved {title}.txt")
    except wikipedia.exceptions.PageError:
        print(f"{title} page not found..... ")

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"{title} is ambiguous.")

    except Exception as e:
        print(f"Skipping '{title}' because of {type(e).__name__}: {e}")

    time.sleep(5)
            
