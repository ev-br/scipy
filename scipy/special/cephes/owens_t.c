/*
 *
 *    	Owen's T function.
 *
 *    Parameters
 *    ----------
 *    h: float64
 *        Input value.
 *    a: float64
 *        Input value.
 *
 *    Returns
 *    -------
 *    t: float64
 *        Probability of the event (X > h and 0 < Y < a * X),
 *        where X and Y are independent standard normal random variables.
 *
 *    Examples
 *    --------
 *    >>> from scipy import special
 *
 *    >>> a = 3.5
 *    >>> h = 0.78
 *    >>> special.owens_t(h, a)
 *    0.10877216734852269
 *
 *    References
 *    ----------
 *    .. [1] M. Patefield and D. Tandy, "Fast and accurate calculation of 
 *           Owen’st function", Statistical Software vol. 5, pp. 1-25, 2000.
 *
 *   
 */ 

#include "mconf.h"
#include <math.h>

static int SELECT_METHOD[] = {
	1, 1, 2, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 9,
    1, 2, 2, 3, 3, 5, 5, 14, 14, 15, 15, 16, 16, 16, 9,
    2, 2, 3, 3, 3, 5, 5, 15, 15, 15, 15, 16, 16, 16, 10,
    2, 2, 3, 5, 5, 5, 5, 7, 7, 16, 16, 16, 16, 16, 10,
    2, 3, 3, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 11,
    2, 3, 5, 5, 5, 6, 6, 8, 8, 17, 17, 17, 12, 12, 12,
    2, 3, 4, 4, 6, 6, 8, 8, 17, 17, 17, 17, 17, 12, 12,
    2, 3, 4, 4, 6, 6, 18, 18, 18, 18, 17, 17, 17, 12, 12};

static double HRANGE[] = {0.2, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6,
    1.7, 2.33, 2.4, 3.36, 3.4, 4.8};

static double ARANGE[] = {0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999};

static double ORD[] = {2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 0, 4, 7,
	8, 20, 0, 0};

static int METHODS[] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4,
	5, 6};

static double C[] = {
	0.99999999999999999999999729978162447266851932041876728736094298092917625009873,
    -0.99999999999999999999467056379678391810626533251885323416799874878563998732905968,
    0.99999999999999999824849349313270659391127814689133077036298754586814091034842536,
    -0.9999999999999997703859616213643405880166422891953033591551179153879839440241685,
    0.99999999999998394883415238173334565554173013941245103172035286759201504179038147,
    -0.9999999999993063616095509371081203145247992197457263066869044528823599399470977,
    0.9999999999797336340409464429599229870590160411238245275855903767652432017766116267,
    -0.999999999574958412069046680119051639753412378037565521359444170241346845522403274,
    0.9999999933226234193375324943920160947158239076786103108097456617750134812033362048,
    -0.9999999188923242461073033481053037468263536806742737922476636768006622772762168467,
    0.9999992195143483674402853783549420883055129680082932629160081128947764415749728967,
    -0.999993935137206712830997921913316971472227199741857386575097250553105958772041501,
    0.99996135597690552745362392866517133091672395614263398912807169603795088421057688716,
    -0.99979556366513946026406788969630293820987757758641211293079784585126692672425362469,
    0.999092789629617100153486251423850590051366661947344315423226082520411961968929483,
    -0.996593837411918202119308620432614600338157335862888580671450938858935084316004769854,
    0.98910017138386127038463510314625339359073956513420458166238478926511821146316469589567,
    -0.970078558040693314521331982203762771512160168582494513347846407314584943870399016019,
    0.92911438683263187495758525500033707204091967947532160289872782771388170647150321633673,
    -0.8542058695956156057286980736842905011429254735181323743367879525470479126968822863,
    0.73796526033030091233118357742803709382964420335559408722681794195743240930748630755,
    -0.58523469882837394570128599003785154144164680587615878645171632791404210655891158,
    0.415997776145676306165661663581868460503874205343014196580122174949645271353372263,
    -0.2588210875241943574388730510317252236407805082485246378222935376279663808416534365,
    0.1375535825163892648504646951500265585055789019410617565727090346559210218472356689,
    -0.0607952766325955730493900985022020434830339794955745989150270485056436844239206648,
    0.0216337683299871528059836483840390514275488679530797294557060229266785853764115,
    -0.00593405693455186729876995814181203900550014220428843483927218267309209471516256,
    0.0011743414818332946510474576182739210553333860106811865963485870668929503649964142,
    -1.489155613350368934073453260689881330166342484405529981510694514036264969925132E-4,
    9.072354320794357587710929507988814669454281514268844884841547607134260303118208E-6 };

static double PTS[] = {
	0.35082039676451715489E-02, 0.31279042338030753740E-01,
    0.85266826283219451090E-01, 0.16245071730812277011E+00,
    0.25851196049125434828E+00, 0.36807553840697533536E+00,
    0.48501092905604697475E+00, 0.60277514152618576821E+00,
    0.71477884217753226516E+00, 0.81475510988760098605E+00,
    0.89711029755948965867E+00, 0.95723808085944261843E+00,
    0.99178832974629703586E+00};

static double WTS[] = {
	0.18831438115323502887E-01, 0.18567086243977649478E-01,
    0.18042093461223385584E-01, 0.17263829606398753364E-01,
    0.16243219975989856730E-01, 0.14994592034116704829E-01,
    0.13535474469662088392E-01, 0.11886351605820165233E-01,
    0.10070377242777431897E-01, 0.81130545742299586629E-02,
    0.60419009528470238773E-02, 0.38862217010742057883E-02,
    0.16793031084546090448E-02};

static double PI = 3.1415926535897932384626433832795;

int get_method(double h, double a) {
	int ihint = 14;
	int iaint = 7;
	int i = 0;

	for (i = 0; i < 14; i++) {
		if (h <= HRANGE[i]) {
			ihint = i;
			break;
		}
	}

	for (i = 0; i < 7; i++) {
		if (h <= HRANGE[i]) {
			ihint = i;
			break;
		}
	}

	return SELECT_METHOD[iaint* 15 + ihint];
}

int get_ord(int index) {
	return ORD[index];
}

double owens_t_norm1(double x) {
    return erf(x / sqrt(2)) / 2;
}

double owens_t_norm2(double x) {
    return erfc(x / sqrt(2)) / 2;
}

double owensT1(double h, double a, double m) {

    double a_sq = a * a;
    double h_sq = -1 * h * h / 2;
    double h_exp = exp(h_sq);

    int i = 0;
    int div = 1;
    double aj = a / (2 * PI);
    double dj = expm1(h_sq);
    double gj = h_sq * h_exp;

    double result = atan(a) * (1 / (2 * PI));

    for (i = 0; i <= m; i++) {
        result += dj * aj / div;
        div += 2;
        aj *= a_sq;
        dj = gj - dj;
        gj *= h_sq / i;
    }

    return result;
}

double owensT2(double h, double a, double m) {

    double h_sq = h * h;
    double ah_sq = -0.5 * h * h * a * a;
    double pi_sq = sqrt(2 * PI);

    int i = 0;
    double z = (ndtr(a * h) - 0.5) / h;

    double result = 0;

    for (i = 0; i <= (2 * m); i++) {
        result += pow(-1, i - 1) * z;
        z = ((2 * i - 1) * z - (pi_sq * pow(a, 2 * i - 1) *
             exp(ah_sq))) / (h_sq);
    }

    result *= -pi_sq * exp(-0.5 * h_sq);

    return result;
}

double owensT3(double h, double a, double m) {
    double a_sq = a * a;
    double h_sq = h * h;
    double y = 1 / h_sq;

    int i = 0;
    double vi = a * exp(- a_sq * h_sq / 2) / sqrt(2 * PI);
    double zi = owens_t_norm1(a * h) / h;
    double result = 0;

    for(i = 0; i<= m; i++) {
        result += zi * C[i];
        zi = y * ((2 * i + 1) * zi - vi);
        vi *= a_sq;
    }

    result *= exp(-h_sq / 2) / sqrt(2 * PI);

    return result;
}

double owensT4(double h, double a, double m) {
    double maxi = 2 * m + 1;
    double h_sq = h * h;
    double a_sq = -a * a;

    int i = 1;
    double ai = a * exp(-h_sq * (1 - a_sq) / 2) / (2 * PI);
    double yi = 1;
    double result = 0;

    while (1) {
        result += ai * yi;

        if (maxi <= i)
            break;

        i += 2;
        yi = (1 - h_sq * yi) / i;
        ai *= a_sq;
    }

    return result;
}

double owensT5(double h, double a, double m) {
	double result = 0;

    double a_sq = a * a;
    double h_sq = -0.5 * h * h;
    int i = 0;

    for (i = 1; i < 14; i++) {
        double r = 1 + a_sq * PTS[i - 1];
        result += WTS[i - 1] * exp(h_sq * r) / r;
    }

    result *= a;

    return result;
}

double owensT6(double h, double a, double m) {

    double result = (0.5 * (ndtr(h)) * (1 - ndtr(h)) - atan((1 - a) /
    	 (1 + a)) * exp(-0.5 * (1 - a) * h * h / atan((1 - a) /
    	 (1 + a))) / (2 * PI));

    return result;
}

double owens_t(double h, double a) {
	if (cephes_isnan(h) || cephes_isnan(a))
		return (NPY_NAN);

    if (h < 0)
        return owens_t(-h, a);

    if (a < 0) {
        return -owens_t(h, a);
    }
    else if (a > 1) {
        double ncdf_h = ndtr(h);
        double ncdf_ah = ndtr(a * h);
        return (0.5 * (ncdf_h + ncdf_ah) - ncdf_h * ncdf_ah -
                owens_t(a * h, 1 / a));
    }

    if (a == 0)
        return 0;

    if (h == 0)
        return 1 * atan(a) / (2 * PI);

    if (a == 1)
        return owens_t_norm2(-h) * owens_t_norm2(h) / 2;

    int index = get_method(h, a);
    int m = get_ord(index);
    int meth_code = METHODS[index - 1];
    double result = 0;

    switch(meth_code) {
    	case 1:
    		result = owensT1(h, a, m);
    		break;
    	case 2:
    		result = owensT2(h, a, m);
    		break;
    	case 3:
    		result = owensT3(h, a, m);
    		break;
    	case 4:
    		result = owensT4(h, a, m);
    		break;
    	case 5:
    		result = owensT5(h, a, m);
    		break;
    	case 6:
    		result = owensT6(h, a, m);
    		break;
    }

    return result;
}
