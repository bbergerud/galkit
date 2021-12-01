"""
Methods for interacting with the SDSS SkyServer SQL database.

Functions
---------
find_nearest_galaxy_objid(ra, dec, radius, dr)
    Finds the objid of the nearest galaxy to the indicated position.

get_photometric_parameters(objid, dr, columns)
    Returns a dictionary containing the PhotoObj parameters
    of the indicated target.

open_sql_search(dr)
    Opens a browser to the SDSS SQL Search page.

open_table_schema(dr)
    Opens a browser to the SDSS Table schema page.

submit_query(query, dr)
    Submits the indicated query to the SDSS SkyServer SQL database.
"""
import io
import mechanicalsoup
import pandas
import webbrowser

def _skyserver_sql_url(dr:int):
    if dr < 10:
        return f'http://cas.sdss.org/DR{dr}/en/tools/search/sql.asp'
    else:
        return f'http://skyserver.sdss.org/dr{dr}/en/tools/search/sql.aspx'

def find_nearest_galaxy_objid(ra:float, dec, radius=1, dr=7):
    """
    Finds the objid of the nearest galaxy to the indicated position.

    Parameters
    ----------
    ra : float
        The right ascension in degrees.

    dec : float
        The declination in degrees.

    radius : float
        The radius in arcminutes over which to search.
    
    dr : int
        The data release number.

    Returns
    -------
    objid : int
        The object id of the nearest galaxy in the indicated data release.

    Examples
    --------
    from galkit.data.sdss.sql import find_nearest_galaxy_objid

    # NGC 0551
    objid = find_nearest_galaxy_objid(21.92, 37.18, dr=7)
    print(objid)
    """
    query = f"""
    SELECT TOP 1 T.objid
    FROM Galaxy as T
    JOIN dbo.fGetNearbyObjEq({ra}, {dec}, {radius}) AS TN
    ON T.objID = TN.objID
    ORDER BY distance
    """
    return submit_query(query, dr=dr).values.item()

def get_photometric_parameters(objid:int, dr:int, columns:list) -> dict:
    """
    Returns a dictionary containing the PhotoObjAll parameters.

    Parameters
    ----------
    objid : int
        The object id value.
    
    dr : int
        The data release number.

    columns : list
        The list of column values to return.
    
    Returns
    -------
    parameters : dict
        A dictionary where the keys are the column names
        and the values are the associated entires in the
        PhotoObj table.

    Examples
    --------
    from galkit.data.sdss.sql import find_nearest_galaxy_objid, get_photometric_parameters

    objid = find_nearest_galaxy_objid(21.92, 37.18, dr=7)
    params = get_photometric_parameters(objid=objid, dr=7, columns=['ra', 'dec', 'petroR90_r'])
    print(params)
    """
    query = f"""
    SELECT {','.join(c for c in columns)}
    FROM PhotoObjAll
    WHERE objid = {objid}
    """
    results = submit_query(query, dr=dr)
    return {k:v for k,v in zip(columns, results.iloc[0].values)}

def open_sql_search(dr:int) -> None:
    """
    Opens a browser to the SDSS SQL Search page.

    Parameters
    ----------
    dr : int
        The data release number.

    Examples
    --------
    from galkit.data.sdss.sql import open_sql_search
    open_sql_search(8)
    """
    webbrowser.open(_skyserver_sql_url(dr)) 

def open_table_schema(table:str, dr:int) -> None:
    """
    Opens a browser to the SDSS Table schema page.

    Parameters
    ----------
    table : str
        The name of the table.

    dr : int
        The data release number.
    
    Examples
    --------
    from galkit.data.sdss.sql import open_table_schema
    open_table_schema('field', 12)
    """
    url = f'http://skyserver.sdss.org/dr{dr}/en/help/browser/browser.asp{"" if dr < 10 else "x"}?n={table}&t=U'
    if dr >= 10:
        url += f'#&&history=description+{table}+U'
    webbrowser.open(url)

def submit_query(query:str, dr:int) -> pandas.DataFrame:
    """
    Submits the indicated query to the SDSS SkyServer SQL database.

    Parameters
    ----------
    query : str
        The SQL query.
    
    dr : int
        The SDSS data release version.

    Returns
    -------
    query : DataFrame
        A pandas dataframe containing the return data.

    Raises
    ------
    Exception
        If the returned response cannot be converted to a pandas
        dataframe, then an exception is raised and the browser
        response is printed. This can happen if the query is invalid.

    Examples
    --------
    from galkit.data.sdss.sql import submit_query

    # Good query
    query = '''SELECT TOP 5 * FROM Galaxy'''
    results = submit_query(query, dr=7)
    print(results)

    # Bad query
    query = '''SELECT * FROM Galaxy LIMIT 5'''
    results = submit_query(query, dr=7)
    """
    # ==========================================================================
    # Establish a connection with the SDSS database
    # ==========================================================================
    browser = mechanicalsoup.StatefulBrowser()
    browser.open(_skyserver_sql_url(dr))
    browser.select_form('#transp form')

    # ==========================================================================
    # Submit the query
    # ==========================================================================
    browser['cmd']    = query
    browser['format'] = 'csv'
    response          = browser.submit_selected()

    # ==========================================================================
    # Extract the query results and return
    #   Issue with #Table1 inserted in later DR releases
    # ==========================================================================
    text = response.text if dr < 10 else response.text[8:]
    try:
        return pandas.read_csv(io.StringIO(text))
    except:
        raise Exception(text)

