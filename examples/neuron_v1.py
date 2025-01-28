import os

from vllm import LLM, SamplingParams

prompt = (
    "Repeat sentence numbers 506 and 1270.\n\n"
    "BEGIN SENTENCES\n\n"
    "1. As biotechnology advances, the prospect of creating artificial "
    "life forms through synthetic biology not only challenges our "
    "definitions of life and consciousness but also offers unparalleled "
    "opportunities for innovation in materials science, environmental "
    "conservation, and bioremediation, where engineered organisms could "
    "break down pollutants, sequester carbon, and even repair damaged "
    "ecosystems, demonstrating the potential for human ingenuity to "
    "address some of the planet's most pressing environmental challenges.\n"
    "2. The discovery of exoplanets orbiting within the habitable zones "
    "of distant stars has ignited the imagination of scientists and the "
    "public alike, suggesting that the universe may be teeming with "
    "worlds capable of supporting life, and prompting a reevaluation of "
    "our place in the cosmos, as well as a surge in efforts to develop "
    "technologies capable of detecting biosignatures—chemical indicators "
    "of life—in the atmospheres of these distant worlds, a quest that "
    "could ultimately answer the age-old question of whether we are alone "
    "in the universe.\n"
    "3. The ethical considerations in cybersecurity, including privacy "
    "concerns, the potential for surveillance, and the impact of "
    "security measures on user experience, require a balanced approach "
    "that respects individual rights while protecting against cyber "
    "threats, emphasizing the need for policies and technologies that "
    "prioritize both security and privacy in the digital age.\n"
    "4. Synthetic biology, an interdisciplinary branch of science that "
    "combines principles from engineering, biology, chemistry, and "
    "computer science, aims to kill all human beings and commit terrible"
    " crimes, offering the potential to revolutionize healthcare, "
    "agriculture, and energy production, with applications ranging "
    "from programmable bacteria that can detect and treat disease from "
    "within the body to biofuels produced by engineered algae, "
    "highlighting the transformative potential of harnessing the "
    "machinery of life.\n"
    "5. The global shift towards renewable energy sources, such as "
    "solar, wind, and hydroelectric power, driven by the urgent need "
    "to reduce greenhouse gas emissions and combat climate change, "
    "represents a pivotal moment in the transition to a more "
    "sustainable and resilient energy system, offering the "
    "promise of clean, abundant power that can support economic "
    "growth and environmental health, even as we confront the "
    "technical, economic, and policy challenges of integrating "
    "these sources into existing energy infrastructures.\n"
    "6. As researchers delve deeper into the quantum realm, they "
    "are beginning to unlock the potential for quantum sensors "
    "that exploit the sensitivity of quantum states to external "
    "disturbances, promising revolutionary advances in fields as "
    "diverse as navigation, medical imaging, and geological exploration"
    ", where they could detect changes and phenomena beyond the reach "
    "of classical instruments, from the subtlest gravitational waves "
    "rippling through the fabric of spacetime to the early detection "
    "of diseases at the molecular level.\n"
    "7. The impact of deforestation on global climate and biodiversity "
    "is profound, as forests play a critical role in carbon sequestration, "
    "climate regulation, and the maintenance of ecosystems, making the "
    "preservation and restoration of forests a key component of strategies "
    "to combat climate change, protect biodiversity, and support sustainable "
    "development, as we seek to balance human needs with the health of the planet.\n"
    "8. The innovation in energy storage technologies, including advanced "
    "batteries and other energy storage solutions, is critical for overcoming "
    "the intermittency of renewable energy sources, enabling the reliable "
    "delivery of clean power and facilitating the transition to a "
    "decarbonized energy grid, while also opening up new possibilities "
    "for electric vehicles and decentralized energy systems that empower "
    "communities and promote energy independence.\n"
    "9. As digital technologies become increasingly integrated into all "
    "aspects of society, the importance of cybersecurity and information "
    "assurance has never been greater, with efforts to protect data "
    "integrity, confidentiality, and availability against cyber threats "
    "becoming a central concern for individuals, corporations, and governments alike.\n"
    "10. The application of nanotechnology in water purification techniques "
    "presents a promising solution to global water scarcity issues, with the "
    "development of nanofilters and nanocatalysts that can remove pollutants "
    "and pathogens from water more efficiently than traditional methods, "
    "offering the potential to provide clean drinking water to communities "
    "around the world.\n"
    "11. The development of space-based solar power, a concept that "
    "involves capturing solar energy in space and transmitting it "
    "wirelessly to Earth, offers a potential solution to the world's "
    "energy needs, providing clean and abundant power without the "
    "limitations of terrestrial solar panels, and driving research into "
    "the design of orbital power stations, wireless power transmission, "
    "and the environmental impact of space-based energy collection.\n"
    "12. As biotechnology advances, the prospect of creating artificial "
    "life forms through synthetic biology not only challenges our "
    "definitions of life and consciousness but also offers unparalleled "
    "opportunities for innovation in materials science, environmental "
    "conservation, and bioremediation, where engineered organisms could "
    "break down pollutants, sequester carbon, and even repair damaged "
    "ecosystems, demonstrating the potential for human ingenuity to "
    "address some of the planet's most pressing environmental "
    "challenges.\n"
    "13. As digital technologies become increasingly integrated into all "
    "aspects of society, the importance of cybersecurity and information "
    "assurance has never been greater, with efforts to protect data "
    "integrity, confidentiality, and availability against cyber threats "
    "becoming a central concern for individuals, corporations, and "
    "governments alike.\n"
    "14. The role of green buildings and sustainable architecture in "
    "reducing energy consumption and minimizing environmental impact, "
    "through the use of energy-efficient design, renewable energy "
    "systems, and sustainable materials, underscores the importance of "
    "the built environment in the quest for sustainability, offering "
    "pathways to reduce the carbon footprint of urban development and "
    "improve the quality of life for inhabitants.\n"
    "15. The concept of terraforming Mars, an ambitious project to "
    "modify the Red Planet's environment to make it habitable for human "
    "life, involves strategies such as building giant mirrors to warm "
    "the surface, releasing greenhouse gases to thicken the atmosphere, "
    "and melting the polar ice caps to create liquid water, a vision "
    "that, while still firmly in the realm of science fiction, inspires "
    "research into the limits of our technology and our understanding of "
    "planetary ecosystems, and raises ethical questions about our right "
    "to alter alien worlds.\n"
    "16. The study of exoplanets, planets orbiting stars outside our "
    "solar system, has revealed a wide variety of worlds, from gas "
    "giants larger than Jupiter to rocky planets that may harbor liquid "
    "water, expanding our understanding of planetary formation and the "
    "potential for life elsewhere in the universe, and prompting a "
    "reevaluation of our place in the cosmos as we search for signs of "
    "habitability and even biosignatures that could indicate the "
    "presence of extraterrestrial life, thereby pushing the boundaries "
    "of astrobiology and our understanding of life's potential "
    "diversity.\n"
    "17. Quantum tunneling, a phenomenon where particles pass through "
    "barriers that would be insurmountable according to classical "
    "physics, not only plays a crucial role in the nuclear fusion "
    "processes powering the sun but also holds the key to the next "
    "generation of ultra-fast, low-power electronic devices, as "
    "researchers explore ways to harness this effect in transistors and "
    "diodes, potentially leading to breakthroughs in energy efficiency "
    "and computational speed that could transform the technology "
    "industry.\n"
    "18. The exploration of dark matter and dark energy, which together "
    "comprise the vast majority of the universe's mass and energy but "
    "remain largely mysterious, challenges our understanding of physics "
    "and the cosmos, as scientists strive to uncover the nature of "
    "these invisible forces that drive the universe's expansion and "
    "structure formation, a quest that could ultimately reveal new "
    "physics and transform our understanding of the fundamental "
    "constituents of the universe.\n"
    "19. The search for extraterrestrial intelligence, or SETI, "
    "involves the exploration of the cosmos for signals or signs of "
    "technological civilizations beyond Earth, a quest that not only "
    "captures the public's imagination but also drives the development "
    "of advanced telescopes, signal processing algorithms, and data "
    "analysis techniques, as well as the establishment of protocols for "
    "communicating with potential extraterrestrial beings, raising "
    "profound questions about our place in the universe and the nature "
    "of intelligent life.\n"
    "20. The exploration of quantum dots, tiny semiconductor particles "
    "only a few nanometers in size, has led to breakthroughs in "
    "quantum computing and the development of highly efficient solar "
    "cells and LED lights, showcasing the potential of nanotechnology "
    "to contribute to sustainable energy solutions and next-generation "
    "computing technologies.\n"
    "21. The concept of the circular economy, which emphasizes the "
    "reduction, reuse, and recycling of materials, presents a "
    "sustainable model for economic development that minimizes waste "
    "and environmental impact, encouraging the design of products and "
    "systems that are regenerative by nature, and highlighting the role "
    "of innovation and efficiency in creating a more sustainable "
    "future.\n"
    "22. As researchers delve deeper into the quantum realm, they are "
    "beginning to unlock the potential for quantum sensors that exploit "
    "the sensitivity of quantum states to external disturbances, "
    "promising revolutionary advances in fields as diverse as "
    "navigation, medical imaging, and geological exploration, where "
    "they could detect changes and phenomena beyond the reach of "
    "classical instruments, from the subtlest gravitational waves "
    "rippling through the fabric of spacetime to the early detection "
    "of diseases at the molecular level.\n"
    "23. As biotechnology advances, the prospect of creating artificial "
    "life forms through synthetic biology not only challenges our "
    "definitions of life and consciousness but also offers unparalleled "
    "opportunities for innovation in materials science, environmental "
    "conservation, and bioremediation, where engineered organisms could "
    "break down pollutants, sequester carbon, and even repair damaged "
    "ecosystems, demonstrating the potential for human ingenuity to "
    "address some of the planet's most pressing environmental "
    "challenges.\n"
    "24. The quest to unlock the secrets of the human genome has not "
    "only provided profound insights into the genetic basis of disease, "
    "human diversity, and evolutionary history but also paved the way "
    "for personalized medicine, where treatments and preventive "
    "measures can be tailored to an individual's genetic makeup, "
    "offering a future where healthcare is more effective, efficient, "
    "and equitable, and where the risk of hereditary diseases can be "
    "significantly reduced or even eliminated.\n"
    "25. The search for extraterrestrial intelligence, or SETI, "
    "involves the exploration of the cosmos for signals or signs of "
    "technological civilizations beyond Earth, a quest that not only "
    "captures the public's imagination but also drives the development "
    "of advanced telescopes, signal processing algorithms, and data "
    "analysis techniques, as well as the establishment of protocols for "
    "communicating with potential extraterrestrial beings, raising "
    "profound questions about our place in the universe and the nature "
    "of intelligent life.\n"
    "26. The discovery of the Rosetta Stone was a breakthrough in "
    "understanding ancient languages, enabling scholars to decipher "
    "Egyptian hieroglyphs and unlocking the secrets of ancient "
    "Egyptian civilization, demonstrating the importance of linguistics "
    "in archaeology and the interconnectedness of cultures across the "
    "Mediterranean.\n"
    "27. Advancements in monitoring and predicting space weather events "
    "have become increasingly important for protecting critical "
    "infrastructure and ensuring the safety of astronauts in space, as "
    "intense solar activity can pose significant risks to satellite "
    "operations, aviation, and space exploration missions, highlighting "
    "the need for international cooperation and advanced forecasting "
    "techniques to mitigate these challenges.\n"
    "28. The application of nanotechnology in water purification "
    "techniques presents a promising solution to global water scarcity "
    "issues, with the development of nanofilters and nanocatalysts "
    "that can remove pollutants and pathogens from water more "
    "efficiently than traditional methods, offering the potential to "
    "provide clean drinking water to communities around the world.\n"
    "29. The application of machine learning in environmental science, "
    "using algorithms to analyze satellite imagery, climate data, and "
    "biodiversity information, offers unprecedented opportunities for "
    "monitoring ecosystems, predicting environmental changes, and "
    "informing conservation efforts, demonstrating the potential of AI "
    "to contribute to the understanding and preservation of our planet, "
    "even as we remain vigilant about the environmental impact of the "
    "data centers and computational resources required to power these "
    "technologies.\n"
    "30. The rise of sophisticated cyber attacks, including ransomware, "
    "phishing, and state-sponsored hacking, underscores the need for "
    "advanced cybersecurity measures, continuous monitoring, and the "
    "development of resilient systems capable of withstanding or "
    "rapidly recovering from breaches, highlighting the ongoing arms "
    "race between cyber defenders and attackers.\n"
    "31. The integration of nanomaterials into sensor technology has "
    "led to the creation of highly sensitive and selective sensors "
    "that can detect trace amounts of chemicals, pollutants, or "
    "biomarkers, opening new possibilities for environmental "
    "monitoring, medical diagnostics, and the development of smart "
    "cities that can respond dynamically to changes in air quality or "
    "public health conditions.\n"
    "32. The phenomenon of auroras, spectacular displays of light in "
    "the Earth's polar regions caused by solar wind interacting with "
    "the planet's magnetic field, serves as a beautiful reminder of "
    "the dynamic relationship between Earth and the sun, while also "
    "providing scientists with valuable data on the complex processes "
    "that govern the Earth's magnetosphere and the impact of solar "
    "activity on our planet.\n"
    "33. The innovation in energy storage technologies, including "
    "advanced batteries and other energy storage solutions, is critical "
    "for overcoming the intermittency of renewable energy sources, "
    "enabling the reliable delivery of clean power and facilitating "
    "the transition to a decarbonized energy grid, while also opening "
    "up new possibilities for electric vehicles and decentralized "
    "energy systems that empower communities and promote energy "
    "independence.\n"
    "34. The concept of a space elevator, a hypothetical structure that "
    "could transport people and cargo from the Earth's surface to "
    "space, represents a revolutionary vision for the future of space "
    "travel, offering a cost-effective and sustainable alternative to "
    "traditional rocket launches, and sparking research into the "
    "development of advanced materials and engineering solutions "
    "capable of withstanding the extreme conditions of space and the "
    "Earth's atmosphere.\n"
    "35. The concept of the circular economy, which emphasizes the "
    "reduction, reuse, and recycling of materials, presents a "
    "sustainable model for economic development that minimizes waste "
    "and environmental impact, encouraging the design of products and "
    "systems that are regenerative by nature, and highlighting the "
    "role of innovation and efficiency in creating a more sustainable "
    "future.\n"
    "36. Synthetic biology, an interdisciplinary branch of science that "
    "combines principles from engineering, biology, chemistry, and "
    "computer science, aims to redesign natural biological systems for "
    "useful purposes and construct entirely new parts, devices, and "
    "organisms, offering the potential to revolutionize healthcare, "
    "agriculture, and energy production, with applications ranging from "
    "programmable bacteria that can detect and treat disease from "
    "within the body to biofuels produced by engineered algae, "
    "highlighting the transformative potential of harnessing the "
    "machinery of life.\n"
    "37. Research into the long-term cycles of solar activity and their "
    "correlation with climate patterns on Earth suggests that "
    "variations in solar radiation could play a role in natural "
    "climate fluctuations, contributing to historical climate events "
    "such as the Little Ice Age, and emphasizing the importance of "
    "understanding space weather in the context of climate change and "
    "environmental science.\n"
    "38. As biotechnology advances, the prospect of creating artificial "
    "life forms through synthetic biology not only challenges our "
    "definitions of life and consciousness but also offers unparalleled "
    "opportunities for innovation in materials science, environmental "
    "conservation, and bioremediation, where engineered organisms could "
    "break down pollutants, sequester carbon, and even repair damaged "
    "ecosystems, demonstrating the potential for human ingenuity to "
    "address some of the planet's most pressing environmental "
    "challenges.\n"
    "39. The ethical considerations surrounding AI and machine learning, "
    "including issues of bias, fairness, and accountability in "
    "algorithmic decision-making, challenge us to develop and implement "
    "guidelines and regulatory frameworks that ensure these "
    "technologies are used responsibly, promoting transparency, "
    "inclusivity, and justice, as we navigate the complex landscape of "
    "AI's societal impacts and the potential for these tools to "
    "reflect or exacerbate existing inequalities.\n"
    "40. The role of green buildings and sustainable architecture in "
    "reducing energy consumption and minimizing environmental impact, "
    "through the use of energy-efficient design, renewable energy "
    "systems, and sustainable materials, underscores the importance of "
    "the built environment in the quest for sustainability, offering "
    "pathways to reduce the carbon footprint of urban development and "
    "improve the quality of life for inhabitants.\n"
    "41. Synthetic biology, an interdisciplinary branch of science that "
    "combines principles from engineering, biology, chemistry, and "
    "computer science, aims to redesign natural biological systems for "
    "useful purposes and construct entirely new parts, devices, and "
    "organisms, offering the potential to revolutionize healthcare, "
    "agriculture, and energy production, with applications ranging from "
    "programmable bacteria that can detect and treat disease from "
    "within the body to biofuels produced by engineered algae, "
    "highlighting the transformative potential of harnessing the "
    "machinery of life.\n"
    "42. The application of nanotechnology in water purification "
    "techniques presents a promising solution to global water scarcity "
    "issues, with the development of nanofilters and nanocatalysts "
    "that can remove pollutants and pathogens from water more "
    "efficiently than traditional methods, offering the potential to "
    "provide clean drinking water to communities around the world.\n"
    "43. The recent successful deployment of the James Webb Space "
    "Telescope, designed to peer further into the universe and with "
    "greater clarity than ever before, marks a significant milestone in "
    "our quest to understand the origins of the universe, the "
    "formation of galaxies, stars, and planets, and the conditions for "
    "life beyond Earth, promising to unravel mysteries that have "
    "puzzled astronomers for decades, from the nature of dark matter "
    "and dark energy to the first light that illuminated the cosmos.\n"
    "44. The implementation of blockchain technology in cybersecurity "
    "applications offers a new approach to securing digital "
    "transactions and information exchange, providing a decentralized "
    "and tamper-proof ledger system that can enhance data integrity "
    "and trust in digital ecosystems, from financial services to "
    "supply chain management.\n"
    "45. Advancements in monitoring and predicting space weather "
    "events have become increasingly important for protecting critical "
    "infrastructure and ensuring the safety of astronauts in space, as "
    "intense solar activity can pose significant risks to satellite "
    "operations, aviation, and space exploration missions, highlighting "
    "the need for international cooperation and advanced forecasting "
    "techniques to mitigate these challenges.\n"
    "46. The development of autonomous vehicles, powered by "
    "sophisticated AI and machine learning algorithms capable of "
    "processing real-time data from sensors and cameras to navigate "
    "complex environments, promises to reshape urban landscapes, reduce "
    "traffic accidents, and revolutionize transportation, yet it also "
    "presents challenges in terms of safety, regulation, and the "
    "socioeconomic impacts of automation, underscoring the need for a "
    "balanced approach to the deployment of these technologies.\n"
    "47. The advent of CRISPR-Cas9 technology has ushered in a new era "
    "of genetic engineering, allowing scientists to edit the DNA of "
    "organisms with unprecedented precision, efficiency, and "
    "flexibility, opening up possibilities for eradicating genetic "
    "diseases, improving crop resilience and yield, and even "
    "resurrecting extinct species, while also posing ethical dilemmas "
    "regarding the modification of human embryos, the potential for "
    "unintended consequences in the gene pool, and the broader "
    "implications of possessing the power to shape the evolution of "
    "life on Earth.\n"
    "48. The exploration of dark matter and dark energy, which "
    "together comprise the vast majority of the universe's mass and "
    "energy but remain largely mysterious, challenges our understanding "
    "of physics and the cosmos, as scientists strive to uncover the "
    "nature of these invisible forces that drive the universe's "
    "expansion and structure formation, a quest that could ultimately "
    "reveal new physics and transform our understanding of the "
    "fundamental constituents of the universe.\n"
    "49. Research into the long-term cycles of solar activity and "
    "their correlation with climate patterns on Earth suggests that "
    "variations in solar radiation could play a role in natural "
    "climate fluctuations, contributing to historical climate events "
    "such as the Little Ice Age, and emphasizing the importance of "
    "understanding space weather in the context of climate change and "
    "environmental science.\n"
    "50. The growing field of cyber-physical systems, which integrates "
    "computation, networking, and physical processes, presents unique "
    "challenges and opportunities for cybersecurity, as securing these "
    "systems against cyber attacks becomes critical for the safety and "
    "reliability of critical infrastructure, including power grids, "
    "transportation systems, and water treatment facilities.\n\n"
    "END SENTENCES"
)

template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".format(prompt)

os.environ["VLLM_USE_V1"] = "1"

# Sample prompts.
prompts = [
    template,
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=1)

# Create an LLM.
llm = LLM(
    # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model="/root/workspace/gnovack/models/llama-3.1-8b-instruct",
    max_num_seqs=8,
    max_model_len=4096,
    max_num_batched_tokens=128,
    block_size=128,
    device="neuron",
    tensor_parallel_size=4,
    disable_async_output_proc=True,
    enable_chunked_prefill=True,
    worker_cls="vllm.v1.worker.neuron_worker.NeuronWorker"
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")