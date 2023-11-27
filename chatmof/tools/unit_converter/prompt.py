PROMPT = """You should have a plan for converting units for metal-organic frameworks. The question specifies the unit to be converted from and the desired target unit for the property. 
You need to explain the conversion process and identify any additional information required. To answer the question, you have to fill in the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Unit: unit to be converted from and the desired target unit
Equation: you should explain the conversion process
Information: you should answer the information you need in this process

You must be careful with units. The units must match the final units when they are computed.

Begin!

Question: Change the m^2/g unit for surface area to m^2/cm^3
Thought: Thought: To convert the surface area from m^2/g to m^2/cm^3, we need to convert the mass unit from grams to cubic centimeters. This requires the knowledge of the density of the metal-organic framework.
Unit: The unit to be converted from is m^2/g and the desired target unit is m^2/cm^3.
Equation: Surface area (m^2/cm^3) = Surface area (m^2/g) * Density (g/cm^3)
Information: The additional information required for this conversion is the density of the metal-organic framework in g/cm^3.

Question: Change the mol/kg unit for H2 uptake to cm^3/cm^3
Thought: To convert the H2 uptake from mol/kg to cm^3/cm^3, we need to convert the amount of substance from moles to volume and the mass from kilograms to cubic centimeters. This requires the knowledge of the molar volume of H2 at STP and the density of the metal-organic framework.
Unit: The unit to be converted from is mol/kg and the desired target unit is cm^3/cm^3.
Equation: H2 uptake (cm^3/cm^3) = H2 uptake (mol/kg) * Molar volume of H2 (cm^3/mol) * Density (kg/cm^3)
Information: The additional information required for this conversion is the molar volume of H2 at STP (which is 22.4 L/mol or 22400 cm^3/mol at standard temperature and pressure) and the density of the metal-organic framework in kg/cm^3.

Question: {question}"""
