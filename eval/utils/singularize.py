import re


VERB, NOUN, ADJECTIVE, ADVERB = "VB", "NN", "JJ", "RB"

plural_prepositions = set((
    "about"  , "before" , "during", "of"   , "till" ,
    "above"  , "behind" , "except", "off"  , "to"   ,
    "across" , "below"  , "for"   , "on"   , "under",
    "after"  , "beneath", "from"  , "onto" , "until",
    "among"  , "beside" , "in"    , "out"  , "unto" ,
    "around" , "besides", "into"  , "over" , "upon" ,
    "at"     , "between", "near"  , "since", "with" ,
    "athwart", "betwixt",
               "beyond",
               "but",
               "by"))

singular_rules = [
    (r'(?i)(.)ae$'            , '\\1a'    ),
    (r'(?i)(.)itis$'          , '\\1itis' ),
    (r'(?i)(.)eaux$'          , '\\1eau'  ),
    (r'(?i)(quiz)zes$'        , '\\1'     ),
    (r'(?i)(matr)ices$'       , '\\1ix'   ),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'   ),
    (r'(?i)^(ox)en'           , '\\1'     ),
    (r'(?i)(alias|status)es$' , '\\1'     ),
    (r'(?i)([octop|vir])i$'   , '\\1us'  ),
    (r'(?i)(cris|ax|test)es$' , '\\1is'   ),
    (r'(?i)(shoe)s$'          , '\\1'     ),
    (r'(?i)(o)es$'            , '\\1'     ),
    (r'(?i)(bus)es$'          , '\\1'     ),
    (r'(?i)([m|l])ice$'       , '\\1ouse' ),
    (r'(?i)(x|ch|ss|sh)es$'   , '\\1'     ),
    (r'(?i)(m)ovies$'         , '\\1ovie' ),
    (r'(?i)(.)ombies$'        , '\\1ombie'),
    (r'(?i)(s)eries$'         , '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'    ),
        # -f, -fe sometimes take -ves in the plural
        # (e.g., lives, wolves).
    (r"([aeo]l)ves$"          , "\\1f"    ),
    (r"([^d]ea)ves$"          , "\\1f"    ),
    (r"arves$"                , "arf"     ),
    (r"erves$"                , "erve"    ),
    (r"([nlw]i)ves$"          , "\\1fe"   ),
    (r'(?i)([lr])ves$'        , '\\1f'    ),
    (r"([aeo])ves$"           , "\\1ve"   ),
    (r'(?i)(sive)s$'          , '\\1'     ),
    (r'(?i)(tive)s$'          , '\\1'     ),
    (r'(?i)(hive)s$'          , '\\1'     ),
    (r'(?i)([^f])ves$'        , '\\1fe'   ),
    # -ses suffixes.
    (r'(?i)(^analy)ses$'      , '\\1sis'  ),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'),
    (r'(?i)(.)opses$'         , '\\1opsis'),
    (r'(?i)(.)yses$'          , '\\1ysis' ),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$', '\\1ose'),
    (r'(?i)(.)oses$'          , '\\1osis' ),
    # -a
    (r'(?i)([ti])a$'          , '\\1um'   ),
    (r'(?i)(n)ews$'           , '\\1ews'  ),
    (r'(?i)s$'                , ''        ),
]

# For performance, compile the regular expressions only once:
singular_rules = [(re.compile(r[0]), r[1]) for r in singular_rules]

singular_uninflected = set((
    "bison"      , "debris"   , "headquarters", "pincers"    , "trout"     ,
    "bream"      , "diabetes" , "herpes"      , "pliers"     , "tuna"      ,
    "breeches"   , "djinn"    , "high-jinks"  , "proceedings", "whiting"   ,
    "britches"   , "eland"    , "homework"    , "rabies"     , "wildebeest",
    "carp"       , "elk"      , "innings"     , "salmon"     ,
    "chassis"    , "flounder" , "jackanapes"  , "scissors"   ,
    "christmas"  , "gallows"  , "mackerel"    , "series"     ,
    "clippers"   , "georgia"  , "measles"     , "shears"     ,
    "cod"        , "graffiti" , "mews"        , "species"    ,
    "contretemps",              "mumps"       , "swine"      ,
    "corps"      ,              "news"        , "swiss"      ,
))
singular_uncountable = set((
    "advice"     , "equipment", "happiness"   , "luggage"    , "news"      , "software"     ,
    "bread"      , "fruit"    , "information" , "mathematics", "progress"  , "understanding",
    "butter"     , "furniture", "ketchup"     , "mayonnaise" , "research"  , "water"        ,
    "cheese"     , "garbage"  , "knowledge"   , "meat"       , "rice"      ,
    "electricity", "gravel"   , "love"        , "mustard"    , "sand"      ,
))
singular_ie = set((
    "alergie"    , "cutie"    , "hoagie"      , "newbie"     , "softie"    , "veggie"       ,
    "auntie"     , "doggie"   , "hottie"      , "nightie"    , "sortie"    , "weenie"       ,
    "beanie"     , "eyrie"    , "indie"       , "oldie"      , "stoolie"   , "yuppie"       ,
    "birdie"     , "freebie"  , "junkie"      , "^pie"       , "sweetie"   , "zombie"       ,
    "bogie"      , "goonie"   , "laddie"      , "pixie"      , "techie"    ,
    "bombie"     , "groupie"  , "laramie"     , "quickie"    , "^tie"      ,
    "collie"     , "hankie"   , "lingerie"    , "reverie"    , "toughie"   ,
    "cookie"     , "hippie"   , "meanie"      , "rookie"     , "valkyrie"  ,
))
singular_irregular = {
       "atlantes": "atlas",
        "atlases": "atlas",
           "axes": "axe",
         "beeves": "beef",
       "brethren": "brother",
       "children": "child",
        "corpora": "corpus",
       "corpuses": "corpus",
    "ephemerides": "ephemeris",
           "feet": "foot",
        "ganglia": "ganglion",
          "geese": "goose",
         "genera": "genus",
          "genii": "genie",
       "graffiti": "graffito",
         "helves": "helve",
           "kine": "cow",
         "leaves": "leaf",
         "loaves": "loaf",
            "men": "man",
      "mongooses": "mongoose",
         "monies": "money",
          "moves": "move",
         "mythoi": "mythos",
         "numena": "numen",
       "occipita": "occiput",
      "octopodes": "octopus",
          "opera": "opus",
         "opuses": "opus",
            "our": "my",
           "oxen": "ox",
          "penes": "penis",
        "penises": "penis",
         "people": "person",
          "sexes": "sex",
    "soliloquies": "soliloquy",
          "teeth": "tooth",
         "testes": "testis",
        "trilbys": "trilby",
         "turves": "turf",
            "zoa": "zoon",
}


def singularize(word, pos=NOUN, custom={}):
    """ Returns the singular of a given word.
    """
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in plural_prepositions:
            return singularize(w[0], pos, custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1]) + "'s"
    w = word.lower()
    for x in singular_uninflected:
        if x.endswith(w):
            return word
    for x in singular_uncountable:
        if x.endswith(w):
            return word
    for x in singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', singular_irregular[x], word)
    for suffix, inflection in singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word