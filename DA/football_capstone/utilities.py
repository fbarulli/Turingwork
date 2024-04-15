from io import StringIO
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET








def extract_xml_details(xml_series):
    """
    Function takes in a series from xml column and returns
    a dataframe with all recorded details
    """

    all_details = []

    for xml_str in xml_series.dropna():
        try:
            root = ET.fromstring(xml_str)

            for value in root.findall("value"):
                detail = {}

                for child in value:
                    if child.tag == "stats":
                        for stat in child:
                            detail[stat.tag] = stat.text
                    else:
                        detail[child.tag] = child.text

                all_details.append(detail)

        except ET.ParseError:
            continue

    return pd.DataFrame(all_details)



def assign_victory(row):
    
    winner = row["home_team_goal"] - row["away_team_goal"]
    
    if winner == 0:
        return "2"
    elif winner < 0:
        return "3"
    else:
        return "1"

def getTeamResult(row):
    
    if row["winning_team"] == "1":
        home_team_result = 'Win'
        away_team_result = 'Loss'
    elif row["winning_team"] == "3":
        home_team_result = 'Loss'
        away_team_result = 'Win'
    else:
        home_team_result = 'Draw'
        away_team_result = 'Draw'
    
    return [home_team_result, away_team_result]




def track_subtype_changes(data):
    subtype_counts = data.groupby(['season', 'subtype']).size().unstack(fill_value=0)
    
    subtype_changes = subtype_counts.diff().abs().sum(axis=0)
    
    most_changed_subtype = subtype_changes.idxmax()
    
    return most_changed_subtype, subtype_changes

def plot_subtype_changes(data, most_changed_subtype):
    most_changed_subtype_data = data[data['subtype'] == most_changed_subtype]
    subtype_counts_per_season = most_changed_subtype_data.groupby('season').size()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subtype_counts_per_season, marker='o', color='skyblue', linewidth=2.5)
    plt.title(f'Changes of Subtype {most_changed_subtype} Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    































































































def try_read_xml(xml_string_or_element):
    """
    Tries to read an XML string or Element object and returns the root element if successful,
    otherwise returns None.
    
    Args:
    - xml_string_or_element (str or Element): The XML string to be parsed or an Element object.
    
    Returns:
    - root (Element): The root element of the parsed XML tree, or None if parsing fails.
    """
    if isinstance(xml_string_or_element, ET.Element):
        return xml_string_or_element  
    try:
        root = ET.fromstring(xml_string_or_element)
        return root
    except ET.ParseError:
        return None

def try_read_xml_series(series):
    """
    Tries to parse XML strings in a Series and returns the root element if successful,
    otherwise returns None.

    Args:
    - series (Series): The Series containing XML strings or XML Element objects.

    Returns:
    - result (Series): Series containing the root elements of parsed XML strings.
    """
    result = series.apply(lambda xml: xml if isinstance(xml, ET.Element) else try_read_xml(xml) if isinstance(xml, str) else None)
    return result

def parse_xml_elements(valid_xml):
    """
    Parses XML elements from a Series into a list of tuples.

    Args:
    - valid_xml (Series): The Series containing valid XML strings.

    Returns:
    - element_data (list): List of tuples containing tag and text pairs.
    """
    valid_xml = valid_xml[valid_xml.notnull()]
    
    element_data = []
    
    for xml_string in valid_xml:
        root = try_read_xml(xml_string)
        if root is not None:
            for element in root.iter():
                element_data.append((element.tag, element.text))
    
    return element_data

def process_element_data(element_data):
    """
    Processes parsed element data into a dictionary.

    Args:
    - element_data (list): List of tuples containing tag and text pairs.

    Returns:
    - result_df (DataFrame): DataFrame containing parsed XML data.
    """
    data = {}
    for tag, text in element_data:
        data.setdefault(tag, []).append(text)

    result_df = create_dataframe_from_data(data)

    return result_df

def create_dataframe_from_data(data):
    """
    Creates a DataFrame from the processed element data.

    Args:
    - data (dict): Dictionary containing processed data grouped by tag.

    Returns:
    - result_df (DataFrame): DataFrame containing parsed XML data.
    """
    max_len = max(len(arr) for key, arr in data.items()) 
    for key in data:
        if len(data[key]) < max_len:  
            data[key] += [np.nan] * (max_len - len(data[key]))

    for key in data:
        data[key] = [float(val) if isinstance(val, str) and val.replace('.', '', 1).isdigit() else val for val in data[key]]

    result_df = pd.DataFrame(data)

    return result_df

def parse_xml_to_dataframe(df, col_name):
    """
    Parses XML strings from a DataFrame column into a DataFrame.

    Args:
    - df (DataFrame): The DataFrame containing the XML strings.
    - col_name (str): The name of the column containing the XML strings.

    Returns:
    - result_df (DataFrame): DataFrame containing parsed XML data.
    """
    valid_xml = try_read_xml_series(df[col_name])  
    valid_xml = valid_xml[valid_xml.notnull()]  
    element_data = parse_xml_elements(valid_xml)
    data = process_element_data(element_data)
    result_df = create_dataframe_from_data(data)
    return result_df