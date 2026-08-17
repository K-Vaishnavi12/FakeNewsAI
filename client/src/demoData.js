// VeriTruth Demo Samples Dataset [PLACEHOLDER / DEMO MODE]
// Pre-configured predictions, SHAP feature attributions, claim decompositions,
// RAG evidence, Google Fact Check Tool records, and technical metadata.

export const DEMO_SAMPLES = [
  {
    id: "demo-sample-1",
    title: "NASA Perseverance Rover Discovers Organic Molecules in Mars Crater",
    category: "Verified Scientific Research",
    verdictLabel: "Likely Real",
    trustScore: 88,
    summary: "Cross-verified with NASA Jet Propulsion Laboratory scientific bulletins and peer-reviewed aerospace releases. Claims regarding organic carbon compounds are corroborated by multiple independent citations.",
    rawText: "NASA's Perseverance rover has discovered diverse organic molecules in the Jezero Crater on Mars. Researchers confirmed the detection of carbon-based compounds using the SHERLOC instrument. Scientists note that while organic molecules contain carbon and hydrogen, they do not constitute definitive proof of past microbial life, requiring further sample return analysis.",
    explanationMethod: "SHAP Feature Attribution (TreeExplainer)",
    shapTokens: [
      { word: "NASA's", weight: 0.42 },
      { word: "Perseverance", weight: 0.38 },
      { word: "rover", weight: 0.25 },
      { word: "has", weight: 0.01 },
      { word: "discovered", weight: 0.15 },
      { word: "diverse", weight: 0.10 },
      { word: "organic", weight: 0.35 },
      { word: "molecules", weight: 0.30 },
      { word: "in", weight: 0.00 },
      { word: "the", weight: 0.00 },
      { word: "Jezero", weight: 0.28 },
      { word: "Crater", weight: 0.22 },
      { word: "on", weight: 0.00 },
      { word: "Mars.", weight: 0.32 },
      { word: "Researchers", weight: 0.20 },
      { word: "confirmed", weight: 0.40 },
      { word: "the", weight: 0.00 },
      { word: "detection", weight: 0.18 },
      { word: "of", weight: 0.00 },
      { word: "carbon-based", weight: 0.29 },
      { word: "compounds", weight: 0.24 },
      { word: "using", weight: 0.05 },
      { word: "the", weight: 0.00 },
      { word: "SHERLOC", weight: 0.45 },
      { word: "instrument.", weight: 0.31 },
      { word: "Scientists", weight: 0.22 },
      { word: "note", weight: 0.10 },
      { word: "that", weight: 0.00 },
      { word: "while", weight: 0.00 },
      { word: "organic", weight: 0.25 },
      { word: "molecules", weight: 0.20 },
      { word: "contain", weight: 0.05 },
      { word: "carbon", weight: 0.15 },
      { word: "and", weight: 0.00 },
      { word: "hydrogen,", weight: 0.12 },
      { word: "they", weight: 0.00 },
      { word: "do", weight: -0.05 },
      { word: "not", weight: -0.08 },
      { word: "constitute", weight: 0.14 },
      { word: "definitive", weight: 0.18 },
      { word: "proof", weight: 0.15 },
      { word: "of", weight: 0.00 },
      { word: "past", weight: 0.05 },
      { word: "microbial", weight: 0.21 },
      { word: "life,", weight: 0.19 },
      { word: "requiring", weight: 0.10 },
      { word: "further", weight: 0.08 },
      { word: "sample", weight: 0.15 },
      { word: "return", weight: 0.18 },
      { word: "analysis.", weight: 0.20 }
    ],
    claims: [
      {
        id: "c1",
        text: "Perseverance detected carbon-based organic compounds using the SHERLOC instrument.",
        status: "Supported",
        statusClass: "status-supported",
        evidence: [
          {
            publisher: "NASA / Jet Propulsion Laboratory",
            title: "Perseverance Identifies Synthesized Organic Material in Jezero",
            snippet: "The SHERLOC instrument on Perseverance detected aromatic organic molecules within the Séítah formation of Jezero Crater.",
            relationship: "Supporting",
            url: "https://www.jpl.nasa.gov/news"
          }
        ]
      },
      {
        id: "c2",
        text: "The discovery confirms the existence of ancient alien life on Mars.",
        status: "Contradicted",
        statusClass: "status-contradicted",
        evidence: [
          {
            publisher: "Nature Astronomy Journal",
            title: "Abiotic Synthesis Paths for Martian Organic Carbon",
            snippet: "Organic molecules can form via non-biological chemical processes such as water-rock interactions and volcanic precipitation.",
            relationship: "Contradicting",
            url: "https://www.nature.com/natastron"
          }
        ]
      }
    ],
    factChecks: [
      {
        publisher: "Google Fact Check / AP News",
        claim: "Organic molecules found on Mars confirm past biological life.",
        rating: "Context Needed / Unproven",
        date: "2024-03-12",
        url: "https://apnews.com/fact-check"
      }
    ],
    technicalDetails: {
      model: "DistilBERT Sequence Classifier",
      modelVersion: "v1.4.2-prod",
      processingTimeMs: 142,
      ragSourcesChecked: 14
    }
  },
  {
    id: "demo-sample-2",
    title: "Breakthrough Battery Charges Electric Vehicles Completely in 30 Seconds",
    category: "Suspicious Tech Claim",
    verdictLabel: "Suspicious",
    trustScore: 54,
    summary: "Contains exaggerated headline metrics. While laboratory pouch cells showed fast energy transfer in micro-tests, full vehicle scale deployment in 30 seconds violates current grid capacity limits and thermal thresholds.",
    rawText: "A revolutionary startup claims to have invented a miracle quantum solid-state battery that charges an entire electric vehicle from 0% to 100% in under 30 seconds without degrading cell life. Industry experts express extreme skepticism, pointing out thermal runaway hazards and lack of peer-reviewed laboratory validation.",
    explanationMethod: "SHAP Feature Attribution (KernelExplainer)",
    shapTokens: [
      { word: "A", weight: 0.0 },
      { word: "revolutionary", weight: -0.32 },
      { word: "startup", weight: -0.15 },
      { word: "claims", weight: -0.40 },
      { word: "to", weight: 0.0 },
      { word: "have", weight: 0.0 },
      { word: "invented", weight: -0.22 },
      { word: "a", weight: 0.0 },
      { word: "miracle", weight: -0.58 },
      { word: "quantum", weight: -0.45 },
      { word: "solid-state", weight: 0.12 },
      { word: "battery", weight: 0.08 },
      { word: "that", weight: 0.0 },
      { word: "charges", weight: -0.10 },
      { word: "an", weight: 0.0 },
      { word: "entire", weight: -0.25 },
      { word: "electric", weight: 0.05 },
      { word: "vehicle", weight: 0.02 },
      { word: "from", weight: 0.0 },
      { word: "0%", weight: -0.18 },
      { word: "to", weight: 0.0 },
      { word: "100%", weight: -0.35 },
      { word: "in", weight: 0.0 },
      { word: "under", weight: -0.15 },
      { word: "30", weight: -0.28 },
      { word: "seconds", weight: -0.30 },
      { word: "without", weight: -0.10 },
      { word: "degrading", weight: 0.05 },
      { word: "cell", weight: 0.02 },
      { word: "life.", weight: 0.01 },
      { word: "Industry", weight: 0.20 },
      { word: "experts", weight: 0.28 },
      { word: "express", weight: 0.15 },
      { word: "extreme", weight: -0.20 },
      { word: "skepticism,", weight: 0.35 },
      { word: "pointing", weight: 0.10 },
      { word: "out", weight: 0.05 },
      { word: "thermal", weight: 0.22 },
      { word: "runaway", weight: 0.18 },
      { word: "hazards", weight: 0.15 },
      { word: "and", weight: 0.0 },
      { word: "lack", weight: -0.30 },
      { word: "of", weight: 0.0 },
      { word: "peer-reviewed", weight: 0.42 },
      { word: "validation.", weight: 0.38 }
    ],
    claims: [
      {
        id: "c1",
        text: "Solid-state battery charges EV completely in under 30 seconds safely.",
        status: "Unverified",
        statusClass: "status-unverified",
        evidence: [
          {
            publisher: "IEEE Spectrum",
            title: "The Physics Limitations of Ultra-Fast Megawatt EV Charging",
            snippet: "Charging a 75kWh battery pack in 30 seconds requires a power delivery rate of 9 Megawatts, exceeding standard grid infrastructure limits.",
            relationship: "Contradicting",
            url: "https://spectrum.ieee.org"
          }
        ]
      }
    ],
    factChecks: [
      {
        publisher: "Snopes Fact Check",
        claim: "Commercial EV battery charges in 30 seconds.",
        rating: "Unproven / Clickbait Title",
        date: "2024-02-18",
        url: "https://www.snopes.com"
      }
    ],
    technicalDetails: {
      model: "DistilBERT Sequence Classifier",
      modelVersion: "v1.4.2-prod",
      processingTimeMs: 195,
      ragSourcesChecked: 8
    }
  },
  {
    id: "demo-sample-3",
    title: "Secret Government Order Mandates Immediate Cancellation of Paper Currency",
    category: "Fabricated Viral Misinformation",
    verdictLabel: "Likely Fake",
    trustScore: 18,
    summary: "High linguistic signals of sensationalism and conspiracy patterns. Fact-checking databases confirm this claim originated from clickbait blogs with zero official treasury backing.",
    rawText: "Urgent leaked memo reveals the central bank has secretly signed an executive order to ban all physical cash starting midnight tonight! Citizens will be forced to surrender paper banknotes immediately or face complete bank account confiscation. Share this warning before it is taken down by censors!",
    explanationMethod: "SHAP Feature Attribution (PartitionExplainer)",
    shapTokens: [
      { word: "Urgent", weight: -0.65 },
      { word: "leaked", weight: -0.55 },
      { word: "memo", weight: -0.30 },
      { word: "reveals", weight: -0.25 },
      { word: "the", weight: 0.0 },
      { word: "central", weight: 0.02 },
      { word: "bank", weight: 0.01 },
      { word: "has", weight: 0.0 },
      { word: "secretly", weight: -0.72 },
      { word: "signed", weight: -0.20 },
      { word: "an", weight: 0.0 },
      { word: "executive", weight: -0.15 },
      { word: "order", weight: -0.20 },
      { word: "to", weight: 0.0 },
      { word: "ban", weight: -0.45 },
      { word: "all", weight: -0.25 },
      { word: "physical", weight: -0.10 },
      { word: "cash", weight: -0.15 },
      { word: "starting", weight: -0.10 },
      { word: "midnight", weight: -0.40 },
      { word: "tonight!", weight: -0.68 },
      { word: "Citizens", weight: -0.20 },
      { word: "will", weight: -0.05 },
      { word: "be", weight: 0.0 },
      { word: "forced", weight: -0.60 },
      { word: "to", weight: 0.0 },
      { word: "surrender", weight: -0.50 },
      { word: "paper", weight: -0.10 },
      { word: "banknotes", weight: -0.15 },
      { word: "immediately", weight: -0.58 },
      { word: "or", weight: 0.0 },
      { word: "face", weight: -0.35 },
      { word: "complete", weight: -0.42 },
      { word: "bank", weight: 0.0 },
      { word: "account", weight: -0.15 },
      { word: "confiscation.", weight: -0.75 },
      { word: "Share", weight: -0.80 },
      { word: "this", weight: -0.25 },
      { word: "warning", weight: -0.62 },
      { word: "before", weight: -0.15 },
      { word: "it", weight: 0.0 },
      { word: "is", weight: 0.0 },
      { word: "taken", weight: -0.30 },
      { word: "down", weight: -0.25 },
      { word: "by", weight: 0.0 },
      { word: "censors!", weight: -0.85 }
    ],
    claims: [
      {
        id: "c1",
        text: "Central bank signed an executive order banning physical cash starting midnight.",
        status: "Contradicted",
        statusClass: "status-contradicted",
        evidence: [
          {
            publisher: "Reuters Fact Check",
            title: "Fact Check: No executive order banning paper cash currency",
            snippet: "The Central Bank confirmed physical currency remains legal tender. Statements claiming midnight cancellation are false and unevidenced.",
            relationship: "Contradicting",
            url: "https://www.reuters.com/fact-check"
          },
          {
            publisher: "AP Fact Check",
            title: "Viral claims of cash confiscation debunked by treasury officials",
            snippet: "Spokesperson confirms no statutory authority exists to arbitrarily seize citizen bank deposits.",
            relationship: "Contradicting",
            url: "https://apnews.com/ap-fact-check"
          }
        ]
      }
    ],
    factChecks: [
      {
        publisher: "Reuters Fact Check",
        claim: "Government secretly signed order banning physical cash.",
        rating: "False",
        date: "2024-01-05",
        url: "https://www.reuters.com/fact-check"
      },
      {
        publisher: "PolitiFact",
        claim: "Paper banknotes banned starting midnight.",
        rating: "Pants on Fire",
        date: "2024-01-06",
        url: "https://www.politifact.com"
      }
    ],
    technicalDetails: {
      model: "DistilBERT Sequence Classifier",
      modelVersion: "v1.4.2-prod",
      processingTimeMs: 110,
      ragSourcesChecked: 22
    }
  }
];

