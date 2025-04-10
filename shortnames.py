"""
Company Name Standardization System
"""

# Dictionary mapping company names to their standardized short forms
shortnames = {
    "AIManufacturingInc": "A&I",
    "ResideoLLC": "ADI",
    "AppliedIndustrialTechnologies": "AIT",
    "AnixterInc": "AXE",
    "BalancingServiceCompanyInc": "BAC",
    "BackflowApparatusValveCo": "Backflow",
    "BearingsandIndustrialSupply": "Bearings",
    "BentleyMillsInc": "Bentley",
    "BuildersHardwareSupplyCoInc": "Builders",
    "ar@cascade-machinery.com": "CascMach",
    "CascadeColumbiaDistribution": "CCD",
    "CEDKENT": "CED",
    "customerservice@cedseattle.com": "CED",
    "CentralStationSteamCo": "CentSS",
    "Billing@centralwelding.com": "CentWeld",
    "Chemaqua": "ChemAqua",
    "CHEMAQUA": "ChemAqua",
    "CINTAS": "Cintas",
    "Cintas_services@cintas.com": "Cintas",
    "CINTASCORP": "Cintas",
    "DorseandCompanyInc": "Dorse",
    "credit@dunnlumber.com": "Dunn",
    "DunnLumberCoInc": "Dunn",
    "EastsideSawSalesInc": "EastSaw",
    "EBBradleyCo": "EBBrad",
    "lily@edensaw.com": "EdenSaw",
    "FastenalCompany": "Fastenal",
    "invoices@fastenal-invoices.com": "Fastenal",
    "ferguson@billtrust.com": "Ferg",
    "FergusonEnterprises": "Ferg",
    "FergusonFireFab": "Ferg",
    "GenscoInc": "Gensco",
    "GlobalIndustrial": "GlobInd",
    "TheGoodyearTireRubberCo": "Goodyear",
    "GraybarElectricCompanyInc": "Graybar",
    "a.white@greatfloors.com": "GreatFloors",
    "GreatFloorsLLC": "GreatFloors",
    "GTSInteriorSupply": "GTS",
    "GuyBrownLLC": "GuyBrown",
    "HarringtonIndustrialPlasticsLLC": "Harrington",
    "HartungGlass": "Hartung",
    "Horizon Distribution, Inc.": "HDI",
    "HDSupplyFacilitiesMaintenanceLtd": "HDSupp",
    "HercRentals": "Herc",
    "HiltiInc": "Hilti",
    "IMLSecuritySupply": "IML",
    "JohnsonControls": "JCI",
    "LWSupplyCorporation": "LWSupp",
    "ar@motors-controls.com": "MCC",
    "McMasterCarrSupplyCompany": "McCarr",
    "MIControls": "MICont",
    "MillersEquipmentRentAllInc": "MillerEquip",
    "ar@mobileelec.com": "MobElec",
    "MorseSteelService": "Morse",
    "pminvoices@PACMAT.COM": "PacMat",
    "PacificPlumbingSupplyCoLLC": "PacPlum",
    "PapeMaterialHandlingInc": "Pape",
    "PartsTownLLC": "PartsT",
    "credit@platt.com": "Platt",
    "PlattElectricSupply": "Platt",
    "PacificOfficeSolutions": "POS",
    "AR@rainiersupply.com": "RainSup",
    "RFIEnterprisesInc": "RFI",
    "RefrigerationSuppliesDistributor": "RSD",
    "customerfinancialservices09@sherwin.com": "Sherwin",
    "TheSherwinWilliamsCo": "Sherwin",
    "SpecialtyDoorServiceInc": "SpecDoor",
    "SteamSupplyLLC": "SteamSupp",
    "StonewayElectricSupply": "Stoneway",
    "TarkettUSAInc": "Tarkett",
    "TASUPPLYCOINCKENTDC": "TASupp",
    "TotalFiltrationServicesInc": "TFS",
    "ThePartsWorks": "TPW",
    "TRAKAUSA": "Traka",
    "ULINE": "Uline",
    "UnitedRefrigerationInc": "UnitedRef",
    "VeritivOperatingCompany": "Veritiv",
}


def process_company_name(extracted_name):
    """
    Process an extracted company name against the shortnames dictionary.

    Parameters:
        extracted_name (str): The company name extracted from a document

    Returns:
        str: The standardized short name if a match is found, or the original name
    """
    # Handle empty or None values
    if not extracted_name or not isinstance(extracted_name, str):
        return "error"

    # Convert to lowercase for case-insensitive matching
    extracted_lower = extracted_name.lower()

    # Check each key in the dictionary for partial matches
    for key in shortnames:
        key_lower = key.lower()
        if key_lower in extracted_lower or extracted_lower in key_lower:
            return shortnames[key]

    # If no match found, return the original name
    return extracted_name
