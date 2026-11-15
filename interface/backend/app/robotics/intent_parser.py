"""Fast deterministic multilingual robot command parser & localized spoken confirmation formatter.

Supports English (en-US/en-GB), Standard German (de-DE/de-CH), Swiss German (gsw),
French (fr-FR/fr-CH), Spanish (es-ES/es-US), Czech (cs-CZ), and Mandarin Chinese (zh-CN/zh-TW).
All parsed intents still pass through AssistantToolExecutor -> RobotSafetyGateway -> ActionSafetyValidator.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# Capability/information questions must go to the conversational LLM rather than triggering motion
_QUESTION_GUARD_RE = re.compile(
    r"(\bhow\s+(far|fast|much|many)\b|\bwhat\s+is\s+the\s+(max|speed|limit|radius)\b"
    r"|\bwie\s+(weit|schnell)\b|\bquelle\s+(vitesse|distance)\b"
    r"|\bcu[áa]n(to)?\s+(r[áa]pido|lejos)\b|\bqu[ée]\s+(velocidad|distancia|alcance)\b|\bcu[áa]l\s+es\s+la\s+velocidad\b"
    r"|\bjak\s+(rychle|daleko)\b|\bjak[áa]\s+je\s+(maxim[áa]ln[íi]|rychlost|vzd[áa]lenost)\b"
    r"|多快|多远|多遠|最大速度|最大半径|最大半徑)",
    re.IGNORECASE,
)

_LOOK_FORWARD_IDIOM_RE = re.compile(
    r"(\blook(ing)?\s+forward\s+to\b|\bsee\s+you\s+(soon|later|tomorrow)\b)",
    re.IGNORECASE,
)

# Visual / camera / object-grounded cues that require triggering `/camera/color/image_raw`
_CAMERA_TRIGGER_RE = re.compile(
    r"(\b(camera|photo|picture|snapshot|image|video|frisbee|disc|bottle|cup|mug|box|ball|obstacle|obstacles"
    r"|object|item|inspect\w*|scan\w*|look\w*|see|seeing|spot|find\w*|locat\w*|search\w*|pick\w*|grab\w*"
    r"|grasp\w*|grip\w*|plac\w*|put\s+down|drop\w*|releas\w*|where\s+is|what\s+(?:do\s+you\s+see|is\s+in\s+front)"
    r"|path\s+clear)\b"
    r"|\b(kamera\w*|foto\w*|bild\w*|scheibe|flasche|kiste|hindernis\w*|objekt\w*|gegenstand|schau\w*"
    r"|sieh\w*|guck\w*|lueg\w*|gseh\w*|prüf\w*|pruef\w*|inspezier\w*|such\w*|suech\w*|find\w*"
    r"|greif\w*|heb\w*|nimm\w*|aufheb\w*|ableg\w*|stell\w*|weg\s+frei)\b"
    r"|\b(cam[ée]ra|bouteille|disque|objet|regarde\w*|vois|voir|inspecte\w*|cherche\w*|trouve\w*"
    r"|attrape\w*|prends|prendre|saisis\w*|pose\w*|rel[âa]che\w*|o[ùu]\s+est)\b"
    r"|\b(c[áa]mara|botella|disco|objeto|obst[áa]culo\w*|mira\w*|ves\b|ver\b|inspeccion\w*|revisa\w*"
    r"|busca\w*|encuentra\w*|agarra\w*|recoge\w*|toma\w*|suelta\w*|coloca\w*|d[óo]nde\s+est[áa])\b"
    r"|\b(kamer\w*|fotk\w*|sn[íi]mek|tal[íi][řr]|l[áa]hev|lahev|p[řr]ek[áa][žz]k\w*|p[řr]edm[ěe]t\w*"
    r"|pod[íi]vej\w*|koukni\w*|vid[íi][šs]|zkontroluj\w*|prohl[ée]dni\w*|prohledni\w*|najdi\w*|hledej\w*"
    r"|uchop\w*|seber\w*|vezmi\w*|polo[žz]\w*|pus[ťt]\w*|kde\s+je)\b"
    r"|(相机|相機|摄像头|攝像頭|拍照|照片|画面|畫面|飞盘|飛盤|瓶|杯|箱|障碍|障礙|物体|物體"
    r"|查看|检查|檢查|观察|觀察|看看|看到|看见|看見|寻找|尋找|找一?下|在哪|抓|拿|捡|撿|放下|松开|鬆開))",
    re.IGNORECASE,
)

# Unmistakable robot-vision / manipulation commands in case LLM omits a tool call
_ROBOT_VISION_FALLBACK_RE = re.compile(
    r"(\b(what\s+(?:do|can)\s+you\s+see|check\s+the\s+(?:camera|workspace|floor|path)|trigger\s+the\s+camera"
    r"|use\s+the\s+camera|look\s+(?:around|at\s+the|for\s+the)|inspect\s+the|scan\s+the\s+(?:room|floor|workspace)"
    r"|find\s+the\s+(?:frisbee|disc|bottle|object|cup|box|ball)|where\s+is\s+the\s+(?:frisbee|disc|bottle|object)"
    r"|pick\s+up\s+the|grab\s+the|grasp\s+the|place\s+the|put\s+down\s+the)\b"
    r"|\b(was\s+siehst\s+du|was\s+gsehsch|schau\s+(?:dich\s+um|nach)|lueg\s+(?:ume|mal)|kamera\s+(?:prüfen|aktivieren|starten)"
    r"|such\w*\s+(?:das|den|die|s|d)\s+(?:frisbee|flasche|scheibe|objekt)|heb\w*\s+.*\bauf|greif\w*\s+(?:das|die|den))\b"
    r"|\b(que\s+vois-tu|qu'est-ce\s+que\s+tu\s+vois|regarde\s+autour|inspecte\s+la|cherche\s+le\s+frisbee"
    r"|attrape\s+la|pose\s+le|pose\s+la)\b"
    r"|\b(qu[ée]\s+ves\b|mira\s+alrededor|busca\s+el\s+frisbee|inspecciona\s+el|recoge\s+la|agarra\s+el|agarra\s+la)\b"
    r"|\b(co\s+(?:te[ďd]\s+)?vid[íi][šs]|pod[íi]vej\s+se\s+kolem|najdi\s+frisbee|zkontroluj\s+(?:prostor|kameru)"
    r"|uchop\s+|seber\s+|polo[žz]\s+)\b"
    r"|(你(?:现在|現在)?看(?:到|见|見)什么|你(?:现在|現在)?看(?:到|见|見)什麼|看看周围|看看周圍|打开相机|打開相機|打开摄像头|打開攝像頭"
    r"|找一?下飞盘|找一?下飛盤|抓起|捡起|撿起|把.*放下))",
    re.IGNORECASE,
)


def should_trigger_robot_camera(text: str) -> bool:
    """Return True when a robotics task requires live visual context from `/camera/color/image_raw`."""
    cleaned = text.strip()
    if not cleaned or _QUESTION_GUARD_RE.search(cleaned) or _LOOK_FORWARD_IDIOM_RE.search(cleaned):
        return False
    return bool(_CAMERA_TRIGGER_RE.search(cleaned))

_ESTOP_RE = re.compile(
    r"(\bemergency\s+stop\b|\be-stop\b|\bestop\b|\bnot-aus\b|\bnotaus\b|\bnotstopp\b"
    r"|\bsofort\s+stopp\w*\b|\barr[êe]t\s+d'urgence\b|\barr[êe]te\s+tout\b"
    r"|\bparada\s+de\s+emergencia\b|\bfreno\s+de\s+emergencia\b|\bdetenci[óo]n\s+de\s+emergencia\b"
    r"|\bnouzov[ée]\s+zastaven[íi]\b|\bnouzov[ýy]\s+stop\b|\bokam[žz]it[ěe]\s+zastav\w*\b"
    r"|急停|紧急停止|緊急停止|立即急停)",
    re.IGNORECASE,
)

_STOP_CMD_RE = re.compile(
    r"(^(please\s+|bitte\s+|s'il\s+vous\s+pla[îi]t\s+|por\s+favor\s+|pros[íi]m\s+|请|請)?"
    r"(stop|halt|freeze|stopp|anhalten|arr[êe]te|arr[êe]tez|para|detente|det[ée]n|alto|zastav|zastavte|st[ůu]j|st[ůu]jte|停|停下|停止|停下来|停下來)[.!?\s¡!。！？]*$"
    r"|\b(stop|halt)\s+(the\s+)?robot\b|\brobot\s+stop\b"
    r"|\broboter\s+(stopp\w*|anhalt\w*|halt)\b|\bhalt\s+de\s+roboter\b"
    r"|\b(arr[êe]te|arr[êe]tez|stoppe|stoppez)\s+le\s+robot\b"
    r"|\b(detente|det[ée]n|parar|frena)\s+el\s+robot\b"
    r"|\b(zastav\w*|st[ůu]j\w*)\s+robota\b"
    r"|机器人停|機器人停|停止移动|停止移動)",
    re.IGNORECASE,
)

_RETURN_HOME_RE = re.compile(
    r"(\b(return|go|move|drive|head|come)\s+(back\s+)?(to\s+)?(home|base|origin|start)\b"
    r"|\bback\s+to\s+(base|home|start|origin)\b"
    r"|\b(zurück|zurueck|zrugg)\s+(zur\s+basis|zum\s+start|zum\s+ursprung|an\s+den\s+anfang|hei)\b"
    r"|\b(geh\w*|fahr\w*|gang|gönd|komm\w*|kehr\w*)\b[^.!?;]{0,25}\b(nach\s+hause|hei|zur\s+basis|zum\s+start)\b"
    r"|\b(nach\s+hause|zur\s+basis|zum\s+start|zum\s+ursprung|hei)\s+(fahren|gehen|zurückkehren|kehren|laufen)\b"
    r"|\b(retourne|retourner|retournez|reviens|revenez|retour)\s+[àa]\s+la\s+(base|maison|position\s+de\s+d[ée]part)\b"
    r"|\bretourne\s+au\s+point\s+de\s+d[ée]part\b"
    r"|\b(vuelve|volver|regresa|regresar|retorna|retornar)\s+(a\s+la\s+base|a\s+casa|al\s+inicio|al\s+origen|al\s+punto\s+de\s+partida)\b"
    r"|\b(vra[ťt]\w*(?:\s+se)?|je[ďd]\w*|jdi|jd[ěe]te|n[áa]vrat)\s+(zp[ěe]t\s+|zp[áa]tky\s+)?(na\s+z[áa]kladnu|dom[ůu]|na\s+za[čc][áa]tek|na\s+start)\b"
    r"|回到起点|回到起點|回起点|回起點|返回基地|返回原点|返回原點|回到原位|回原点|回原點)",
    re.IGNORECASE,
)

_LINEAR_UNIT_PAT = (
    r"(?:mm|millimeters?|millimetres?|millimeter|millim[èe]tres?|mil[íi]metros?|milimetr\w*"
    r"|cm|centimeters?|centimetres?|zentimeter|centim[èe]tres?|cent[íi]metros?|centimetr\w*"
    r"|ft|feet|foot|inch|inches|m|meters?|metres?|meter|m[èe]tres?|metros?|metr\w*"
    r"|毫米|厘米|公分|米|公尺)"
    r"(?:\s+(?:and\s+a\s+half|et\s+demie?|y\s+medi[oa]|a\s+p[ůu]l))?"
)

_ANGLE_UNIT_PAT = (
    r"(?:°|deg(?:rees?)?|rad(?:ians?)?|radiant\w*|radianes|radi[áa]n\w*"
    r"|grad|degr[ée]s?|grados?|stup[ňně]\w*|stupen|度|弧度)"
)

_ROTATE_RE = re.compile(
    r"(\b(turn|rotate|spin|pivot)\b[^.!?;]{0,40}\b(around|left|right|clockwise|counter-?clockwise|anti-?clockwise|\d|degrees?|deg\b|radians?|rad\b)"
    r"|\b(dreh\w*|wend\w*|träi\w*|dräi\w*)\b.*?\b(links|rechts|rächts|um|ume|\d|grad\b)"
    r"|\b(umdreh\w*|umdräi\w*|kehrtmach\w*|kehrt\s+machen)\b"
    r"|\b(nach\s+links|nach\s+rechts|nach\s+rächts|links|rechts|rächts|um\s+\d+\s*(?:grad|°)?|\d+\s*(?:grad|°))\b[^.!?;]{0,20}\b(dreh\w*|wend\w*|träi\w*|dräi\w*)\b"
    r"|\b(tourne|tourner|tournez|pivote|pivotez)\b.*?\b(gauche|droite|demi-tour|\d|degr[ée]s?)"
    r"|\bfais\s+demi-tour\b"
    r"|\b(gira\w*|rota\w*|voltea\w*)\b.*?\b(izquierda|derecha|vuelta|redondo|\d|grados?)"
    r"|\b(date\s+la\s+vuelta|da\s+media\s+vuelta|media\s+vuelta)\b"
    r"|\b(oto[čc]\w*|zato[čc]\w*)\b.*?\b(doleva|vlevo|doprava|vpravo|se|kolem|dokola|vzad|\d|stup)"
    r"|\b([čc]elem\s+vzad|(?:doleva|vlevo|doprava|vpravo)\s+(?:se\s+)?(?:oto[čc]\w*|zato[čc]\w*))\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|ninety|forty[-\s]?five|thirty|sixty|fünfundvierzig|neunzig|quarante-cinq|quatre-vingt-dix|nonante|cuarenta\s+y\s+cinco|noventa|[čc]ty[řr]icet\s+p[ěe]t|devades[áa]t)\s*{_ANGLE_UNIT_PAT}\s*(?:to\s+the\s+|nach\s+|vers\s+la\s+|[àa]\s+(?:la\s+)?|hacia\s+la\s+)?\b(left|right|clockwise|counter-?clockwise|anti-?clockwise|links|rechts|rächts|gauche|droite|izquierda|derecha|doleva|vlevo|doprava|vpravo)\b"
    rf"|\b(left|right|clockwise|counter-?clockwise|anti-?clockwise|links|rechts|rächts|gauche|droite|izquierda|derecha|doleva|vlevo|doprava|vpravo)\s+(?:by\s+|um\s+|de\s+|o\s+)?\d+(?:[.,]\d+)?\s*{_ANGLE_UNIT_PAT}"
    r"|(?:向左|往左|朝左|向右|往右|朝右)\s*(?:\d+(?:[.,]\d+)?|[一二两兩三四五六七八九十百]+)\s*(?:度|弧度)"
    r"|(?:\d+(?:[.,]\d+)?|[一二两兩三四五六七八九十百]+)\s*(?:度|弧度)\s*(?:向左|往左|向右|往右)"
    r"|向左转|往左转|左转|向左轉|往左轉|左轉|向右转|往右转|右转|向右轉|往右轉|右轉|转身|轉身|掉头|掉頭|旋转|旋轉)",
    re.IGNORECASE,
)

_FORWARD_RE = re.compile(
    r"(\b(go|move|drive|walk|step|advance|head|proceed|travel)\b[^.!?;]{0,35}\b(forward|forwards|ahead|straight)\b"
    r"|^(please\s+)?(forward|forwards|ahead|advance|straight\s+ahead)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|[a-zA-Z]+(?:\s+(?:point|and\s+a)\s+[a-zA-Z]+)?)\s*{_LINEAR_UNIT_PAT}\s+(forward|forwards|ahead|straight(?:\s+ahead)?|en\s+avant|tout\s+droit)\b"
    rf"|\b(forward|forwards|ahead|straight(?:\s+ahead)?|en\s+avant|tout\s+droit)\s+(?:by\s+|de\s+)?(?:\d+(?:[.,]\d+)?|half|one|two|three|four|five|six|seven|eight|nine|ten|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s*{_LINEAR_UNIT_PAT}\b"
    r"|\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd|chumm)\b[^.!?;]{0,35}\b(vorwärts|vorwaerts|nach\s+vorne?|geradeaus|graduus|füre|fürä|vüre)\b"
    r"|\b(vorwärts|vorwaerts|nach\s+vorne?|geradeaus|graduus|füre|fürä|vüre)\b[^.!?;]{0,25}\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|halb\w*|ein\w*|anderthalb|zweieinhalb|dreieinhalb|zwei|drei|vier|fünf|fuenf|sechs|sieben|acht|neun|zehn|zwanzig|füüf|zwöi|drü)\s*{_LINEAR_UNIT_PAT}\s+(vorwärts|vorwaerts|nach\s+vorne?|geradeaus|graduus|füre|fürä|vüre)\b"
    rf"|\b(vorwärts|vorwaerts|nach\s+vorne?|geradeaus|graduus|füre|fürä|vüre)\s+(?:um\s+)?(?:\d+(?:[.,]\d+)?|halb\w*|ein\w*|anderthalb|zwei|drei|vier|fünf|fuenf|füüf|zwöi|drü)\s*{_LINEAR_UNIT_PAT}\b"
    r"|^(bitte\s+)?(vorwärts|vorwaerts|füre|fürä|vüre|graduus)\b"
    r"|\b(avance|avancer|avancez)\b"
    r"|\b(va|allez|marche|roule|d[ée]place[z-]?vous|d[ée]place)\b[^.!?;]{0,30}\b(en\s+avant|tout\s+droit)\b"
    r"|\b(avanza|avanzar|adelanta)\b"
    r"|\b(ve|camina|anda|mu[ée]vete|despl[áa]zate|conduce|dir[íi]gete)\b[^.!?;]{0,35}\b(adelante|hacia\s+adelante|al\s+frente|recto|todo\s+recto|derecho)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|medi[oa]|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|veinte)\s*{_LINEAR_UNIT_PAT}\s+(adelante|hacia\s+adelante|al\s+frente|recto)\b"
    rf"|\b(adelante|hacia\s+adelante|al\s+frente|todo\s+recto)\s+(?:\d+(?:[.,]\d+)?|medi[oa]|un|uno|una|dos|tres|cuatro|cinco|diez)\s*{_LINEAR_UNIT_PAT}\b"
    r"|\b(jdi|jd[ěe]te|je[ďd]\w*|popoje[ďd]\w*|posu[ňn]\w*|vykro[čc]\w*|pokra[čc]uj\w*)\b[^.!?;]{0,35}\b(dop[řr]edu|vp[řr]ed|rovn[ěe]|kup[řr]edu)\b"
    r"|\b(dop[řr]edu|vp[řr]ed|rovn[ěe]|kup[řr]edu)\b[^.!?;]{0,25}\b(jdi|jd[ěe]te|je[ďd]\w*|popoje[ďd]\w*|posu[ňn]\w*)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|p[ůu]l|jeden|jednu|jedna|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t|[šs]est|sedm|osm|dev[ěe]t|deset|dvacet)\s*{_LINEAR_UNIT_PAT}\s+(dop[řr]edu|vp[řr]ed|rovn[ěe]|kup[řr]edu)\b"
    rf"|\b(dop[řr]edu|vp[řr]ed|rovn[ěe]|kup[řr]edu)\s+(?:o\s+)?(?:\d+(?:[.,]\d+)?|p[ůu]l|jeden|jednu|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t|deset)\s*{_LINEAR_UNIT_PAT}\b"
    r"|^(pros[íi]m\s+)?(dop[řr]edu|vp[řr]ed|kup[řr]edu)\b"
    r"|(向前|往前|朝前|前进|前進|直行|直走)[^。！？.!?;]{0,6}(走|移动|移動|行|开|開|跑|米|公尺|厘米|公分|毫米|步|\d|[零〇一二两兩三四五六七八九十半])"
    r"|(?:\d+(?:[.,]\d+)?|[零〇半一二两兩三四五六七八九十点點]+)\s*(?:毫米|厘米|公分|米|公尺)半?\s*(?:向前|往前|朝前|前进|前進|直行|直走)"
    r"|^(请|請)?(前进|前進|往前走|向前走)[。！？.!?\s]*$)",
    re.IGNORECASE,
)

_BACKWARD_RE = re.compile(
    r"(\b(go|move|drive|walk|step|reverse)\b[^.!?;]{0,35}\b(backward|backwards|back)\b"
    r"|\bback\s+up\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|[a-zA-Z]+(?:\s+(?:point|and\s+a)\s+[a-zA-Z]+)?)\s*{_LINEAR_UNIT_PAT}\s+(backward|backwards|back|en\s+arri[èe]re)\b"
    rf"|\b(backward|backwards|back|reverse|en\s+arri[èe]re)\s+(?:by\s+|de\s+)?(?:\d+(?:[.,]\d+)?|half|demie?|one|two|three|four|five|six|seven|eight|nine|ten|un|une|deux|trois|quatre|cinq)\s*{_LINEAR_UNIT_PAT}\b"
    r"|\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b[^.!?;]{0,35}\b(rückwärts|rueckwaerts|zurück|zurueck|nach\s+hinten|zrugg|hinderschi|nach\s+hinde)\b"
    r"|\b(rückwärts|rueckwaerts|nach\s+hinten|zrugg|hinderschi)\b[^.!?;]{0,25}\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|halb\w*|ein\w*|anderthalb|zweieinhalb|zwei|drei|vier|fünf|fuenf|sechs|sieben|acht|neun|zehn|füüf|zwöi|drü)\s*{_LINEAR_UNIT_PAT}\s+(rückwärts|rueckwaerts|zurück|zurueck|nach\s+hinten|zrugg|hinderschi)\b"
    rf"|\b(rückwärts|rueckwaerts|zurück|zurueck|nach\s+hinten|zrugg|hinderschi)\s+(?:um\s+)?(?:\d+(?:[.,]\d+)?|halb\w*|ein\w*|anderthalb|zwei|drei|vier|fünf|fuenf|füüf|zwöi|drü)\s*{_LINEAR_UNIT_PAT}\b"
    r"|\b(recule|reculer|reculez)\b"
    r"|\b(va|allez|marche|roule)\b[^.!?;]{0,30}\b(en\s+arri[èe]re)\b"
    r"|\b(retrocede|retroceder)\b"
    r"|\b(ve|camina|anda|mu[ée]vete|despl[áa]zate|marcha)\b[^.!?;]{0,35}\b(atr[áa]s|hacia\s+atr[áa]s|para\s+atr[áa]s)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|medi[oa]|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s*{_LINEAR_UNIT_PAT}\s+(atr[áa]s|hacia\s+atr[áa]s)\b"
    rf"|\b(atr[áa]s|hacia\s+atr[áa]s)\s+(?:\d+(?:[.,]\d+)?|medi[oa]|un|uno|una|dos|tres|cuatro|cinco)\s*{_LINEAR_UNIT_PAT}\b"
    r"|\b(couvej\w*|couvni\w*|couvn[ěe]te)\b"
    r"|\b(jdi|jd[ěe]te|je[ďd]\w*|posu[ňn]\w*)\b[^.!?;]{0,35}\b(dozadu|zp[ěe]t|nazp[ěe]t|zp[áa]tky|vzad)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|p[ůu]l|jeden|jednu|jedna|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t|[šs]est|sedm|osm|dev[ěe]t|deset)\s*{_LINEAR_UNIT_PAT}\s+(dozadu|zp[ěe]t|zp[áa]tky)\b"
    rf"|\b(dozadu|zp[ěe]t|zp[áa]tky)\s+(?:o\s+)?(?:\d+(?:[.,]\d+)?|p[ůu]l|jeden|jednu|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t)\s*{_LINEAR_UNIT_PAT}\b"
    r"|(?:\d+(?:[.,]\d+)?|[零〇半一二两兩三四五六七八九十点點]+)\s*(?:毫米|厘米|公分|米|公尺)半?\s*(?:向后|往后|向後|往後|后退|後退|倒退)"
    r"|(向后|往后|向後|往後|后退|後退|倒退))",
    re.IGNORECASE,
)

_STRAFE_LEFT_RE = re.compile(
    r"(\b(go|move|strafe|slide|step|shift)\b[^.!?;]{0,30}\b(left)\b"
    r"|\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b[^.!?;]{0,30}\b(nach\s+links|links)\b"
    r"|\b(nach\s+links)\b[^.!?;]{0,20}\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b"
    r"|\b(va|allez|d[ée]place|glisse)\b[^.!?;]{0,30}\b([àa]\s+gauche|vers\s+la\s+gauche)\b"
    r"|\b(ve|camina|mu[ée]vete|despl[áa]zate)\b[^.!?;]{0,30}\b(a\s+la\s+izquierda|hacia\s+la\s+izquierda)\b"
    r"|\b(jdi|jd[ěe]te|je[ďd]\w*|posu[ňn]\w*)\b[^.!?;]{0,30}\b(doleva|vlevo)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|half|halb\w*|demie?|medi[oa]|p[ůu]l|one|two|three|four|five|six|seven|eight|nine|ten|ein\w*|anderthalb|zwei|drei|vier|fünf|fuenf|füüf|zwöi|drü|un|une|deux|trois|quatre|cinq|uno|una|dos|tres|cuatro|cinco|jeden|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t)\s*{_LINEAR_UNIT_PAT}\s+(?:to\s+the\s+|nach\s+|vers\s+la\s+|[àa]\s+(?:la\s+)?|hacia\s+la\s+)?(left|links|gauche|izquierda|doleva|vlevo)\b"
    rf"|\b(?:to\s+the\s+|nach\s+|[àa]\s+|a\s+la\s+|hacia\s+la\s+)?(left|links|gauche|izquierda|doleva|vlevo)\s+(?:by\s+|um\s+|de\s+|o\s+)?(?:\d+(?:[.,]\d+)?|half|halb\w*|demie?|medi[oa]|p[ůu]l|one|two|three|four|five|ein\w*|zwei|drei|vier|fünf|un|une|deux|trois|uno|dos|tres|jeden|dva|t[řr]i)\s*{_LINEAR_UNIT_PAT}\b"
    r"|(?:\d+(?:[.,]\d+)?|[零〇半一二两兩三四五六七八九十点點]+)\s*(?:毫米|厘米|公分|米|公尺)半?\s*(?:向左|往左|朝左|左移)"
    r"|(向左|往左|朝左|左移)[^。！？.!?;]{0,6}(走|移动|移動|平移|米|公尺|厘米|公分|毫米|步|\d|[零〇一二两兩三四五六七八九十半]))",
    re.IGNORECASE,
)

_STRAFE_RIGHT_RE = re.compile(
    r"(\b(go|move|strafe|slide|step|shift)\b[^.!?;]{0,30}\b(right)\b"
    r"|\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b[^.!?;]{0,30}\b(nach\s+rechts|nach\s+rächts|rechts|rächts)\b"
    r"|\b(nach\s+rechts|nach\s+rächts)\b[^.!?;]{0,20}\b(geh\w*|fahr\w*|beweg\w*|lauf\w*|gang|gönd)\b"
    r"|\b(va|allez|d[ée]place|glisse)\b[^.!?;]{0,30}\b([àa]\s+droite|vers\s+la\s+droite)\b"
    r"|\b(ve|camina|mu[ée]vete|despl[áa]zate)\b[^.!?;]{0,30}\b(a\s+la\s+derecha|hacia\s+la\s+derecha)\b"
    r"|\b(jdi|jd[ěe]te|je[ďd]\w*|posu[ňn]\w*)\b[^.!?;]{0,30}\b(doprava|vpravo)\b"
    rf"|\b(?:\d+(?:[.,]\d+)?|half|halb\w*|demie?|medi[oa]|p[ůu]l|one|two|three|four|five|six|seven|eight|nine|ten|ein\w*|anderthalb|zwei|drei|vier|fünf|fuenf|füüf|zwöi|drü|un|une|deux|trois|quatre|cinq|uno|una|dos|tres|cuatro|cinco|jeden|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t)\s*{_LINEAR_UNIT_PAT}\s+(?:to\s+the\s+|nach\s+|vers\s+la\s+|[àa]\s+(?:la\s+)?|hacia\s+la\s+)?(right|rechts|rächts|droite|derecha|doprava|vpravo)\b"
    rf"|\b(?:to\s+the\s+|nach\s+|[àa]\s+|a\s+la\s+|hacia\s+la\s+)?(right|rechts|rächts|droite|derecha|doprava|vpravo)\s+(?:by\s+|um\s+|de\s+|o\s+)?(?:\d+(?:[.,]\d+)?|half|halb\w*|demie?|medi[oa]|p[ůu]l|one|two|three|four|five|ein\w*|zwei|drei|vier|fünf|un|une|deux|trois|uno|dos|tres|jeden|dva|t[řr]i)\s*{_LINEAR_UNIT_PAT}\b"
    r"|(?:\d+(?:[.,]\d+)?|[零〇半一二两兩三四五六七八九十点點]+)\s*(?:毫米|厘米|公分|米|公尺)半?\s*(?:向右|往右|朝右|右移)"
    r"|(向右|往右|朝右|右移)[^。！？.!?;]{0,6}(走|移动|移動|平移|米|公尺|厘米|公分|毫米|步|\d|[零〇一二两兩三四五六七八九十半]))",
    re.IGNORECASE,
)

_TORSO_LIFT_RE = re.compile(
    r"\b(raise|lift|elevate|lower|set)\s+(?:the\s+)?(?:robot(?:'s)?\s+|panther(?:'s)?\s+)?(torso|lift(?:\s+column)?|mast|body\s+height)\b",
    re.IGNORECASE,
)

_SPOKEN_DIGIT_MAP: dict[str, int] = {
    "zero": 0, "oh": 0, "null": 0, "zéro": 0, "cero": 0, "nula": 0,
    "one": 1, "ein": 1, "eins": 1, "eine": 1, "einen": 1, "un": 1, "une": 1, "uno": 1, "una": 1, "jeden": 1, "jedna": 1, "jednu": 1,
    "two": 2, "zwei": 2, "zwöi": 2, "deux": 2, "dos": 2, "dva": 2, "dvě": 2, "dve": 2,
    "three": 3, "drei": 3, "drü": 3, "drüü": 3, "trois": 3, "tres": 3, "tři": 3, "tri": 3,
    "four": 4, "vier": 4, "quatre": 4, "cuatro": 4, "čtyři": 4, "ctyri": 4,
    "five": 5, "fünf": 5, "fuenf": 5, "füf": 5, "füüf": 5, "cinq": 5, "cinco": 5, "pět": 5, "pet": 5,
    "six": 6, "sechs": 6, "sächs": 6, "seis": 6, "šest": 6, "sest": 6,
    "seven": 7, "sieben": 7, "sibe": 7, "sept": 7, "siete": 7, "sedm": 7,
    "eight": 8, "acht": 8, "huit": 8, "ocho": 8, "osm": 8,
    "nine": 9, "neun": 9, "nüün": 9, "neuf": 9, "nueve": 9, "devět": 9, "devet": 9,
    "ten": 10, "zehn": 10, "zäh": 10, "dix": 10, "diez": 10, "deset": 10,
}

_GERMAN_COMPOUND_HALVES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(anderthalb|eineinhalb)\b", re.I), 1.5),
    (re.compile(r"\bzweieinhalb\b", re.I), 2.5),
    (re.compile(r"\bdreieinhalb\b", re.I), 3.5),
    (re.compile(r"\bviereinhalb\b", re.I), 4.5),
    (re.compile(r"\b(fünfeinhalb|fuenfeinhalb)\b", re.I), 5.5),
    (re.compile(r"\bsechseinhalb\b", re.I), 6.5),
    (re.compile(r"\bsiebeneinhalb\b", re.I), 7.5),
    (re.compile(r"\bachteinhalb\b", re.I), 8.5),
    (re.compile(r"\bneuneinhalb\b", re.I), 9.5),
    (re.compile(r"\bzehneinhalb\b", re.I), 10.5),
)

_SPOKEN_AND_HALF_RE = re.compile(
    r"\b(?P<base>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix"
    r"|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez"
    r"|jeden|jedna|jednu|dva|dv[ěe]|t[řr]i|[čc]ty[řr]i|p[ěe]t|[šs]est|sedm|osm|dev[ěe]t|deset)"
    r"\s+(?:(?:m|meters?|metres?|m[èe]tres?|metros?|metr\w*)\s+)?"
    r"(?:and\s+a\s+half|et\s+demie?|y\s+medi[oa]|a\s+p[ůu]l)\b",
    re.IGNORECASE,
)

_SPOKEN_DECIMAL_RE = re.compile(
    r"\b(?P<whole>\d+|zero|null|z[ée]ro|cero|nula|one|ein\w*|un|une|uno|una|jeden|jedna|two|zwei|zwöi|deux|dos|dva|dv[ěe]"
    r"|three|drei|drü|trois|tres|t[řr]i|four|vier|quatre|cuatro|[čc]ty[řr]i|five|fünf|fuenf|füüf|cinq|cinco|p[ěe]t"
    r"|six|sechs|seis|[šs]est|seven|sieben|sept|siete|sedm|eight|acht|huit|ocho|osm|nine|neun|neuf|nueve|dev[ěe]t|ten|zehn|dix|diez|deset)"
    r"\s+(?:point|dot|komma|virgule|punto|coma|cel[áa]|cel[ýy]ch)\s+"
    r"(?P<frac>\d+|twenty[-\s]?five|seventy[-\s]?five|zero|oh|null|z[ée]ro|cero|nula|one|eins?|un|uno|jeden"
    r"|two|zwei|deux|dos|dva|three|drei|trois|tres|t[řr]i|four|vier|quatre|cuatro|[čc]ty[řr]i"
    r"|five|fünf|fuenf|cinq|cinco|p[ěe]t|six|sechs|seis|[šs]est|seven|sieben|sept|siete|sedm"
    r"|eight|acht|huit|ocho|osm|nine|neun|neuf|nueve|dev[ěe]t)\b",
    re.IGNORECASE,
)

_WORD_NUMBERS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(three\s+hundred(?:\s+and)?\s+sixty|dreihundertsechzig|trois\s+cent\s+soixante|trescientos\s+sesenta|t[řr]i\s+sta\s+[šs]edes[áa]t)\b", re.I), 360.0),
    (re.compile(r"\b(two\s+hundred(?:\s+and)?\s+seventy|zweihundertsiebzig|deux\s+cent\s+soixante-dix|doscientos\s+setenta|dv[ěe]\s+st[ěe]\s+sedmdes[áa]t)\b", re.I), 270.0),
    (re.compile(r"\b(one\s+hundred(?:\s+and)?\s+eighty|hundred\s+and\s+eighty|one\s+eighty|hundertachtzig|cent\s+quatre-vingts?|ciento\s+ochenta|sto\s+osmdes[áa]t)\b", re.I), 180.0),
    (re.compile(r"(一百八十|一百八)"), 180.0),
    (re.compile(r"\b(one\s+hundred(?:\s+and)?\s+fifty|hundertfünfzig|cent\s+cinquante|ciento\s+cincuenta|sto\s+pades[áa]t)\b", re.I), 150.0),
    (re.compile(r"(一百五十|一百五)"), 150.0),
    (re.compile(r"\b(one\s+hundred(?:\s+and)?\s+thirty[-\s]?five|hundertfünfunddreissig|hundertfünfunddreißig|cent\s+trente-cinq|ciento\s+treinta\s+y\s+cinco|sto\s+t[řr]icet\s+p[ěe]t)\b", re.I), 135.0),
    (re.compile(r"(一百三十五)"), 135.0),
    (re.compile(r"\b(one\s+hundred(?:\s+and)?\s+twenty|hundertzwanzig|cent\s+vingt|ciento\s+veinte|sto\s+dvacet)\b", re.I), 120.0),
    (re.compile(r"(一百二十|一百二)"), 120.0),
    (re.compile(r"\b(one\s+hundred|a\s+hundred|einhundert|hundert|cent\b|cien|ciento|sto)\b", re.I), 100.0),
    (re.compile(r"(一百)"), 100.0),
    (re.compile(r"\b(ninety|neunzig|nünzg|quatre-vingt-dix|nonante|noventa|devades[áa]t)\b", re.I), 90.0),
    (re.compile(r"(九十)"), 90.0),
    (re.compile(r"\b(eighty|achtzig|quatre-vingts?|huitante|octante|ochenta|osmdes[áa]t)\b", re.I), 80.0),
    (re.compile(r"(八十)"), 80.0),
    (re.compile(r"\b(seventy[-\s]?five|fünfundsiebzig|soixante-quinze|septante-cinq|setenta\s+y\s+cinco|sedmdes[áa]t\s+p[ěe]t)\b", re.I), 75.0),
    (re.compile(r"(七十五)"), 75.0),
    (re.compile(r"\b(seventy|siebzig|soixante-dix|septante|setenta|sedmdes[áa]t)\b", re.I), 70.0),
    (re.compile(r"(七十)"), 70.0),
    (re.compile(r"\b(sixty|sechzig|soixante|sesenta|[šs]edes[áa]t)\b", re.I), 60.0),
    (re.compile(r"(六十)"), 60.0),
    (re.compile(r"\b(fifty|fünfzig|fuenfzig|cinquante|cincuenta|pades[áa]t)\b", re.I), 50.0),
    (re.compile(r"(五十)"), 50.0),
    (re.compile(r"\b(forty[-\s]?five|fünfundvierzig|quarante-cinq|cuarenta\s+y\s+cinco|[čc]ty[řr]icet\s+p[ěe]t)\b", re.I), 45.0),
    (re.compile(r"(四十五)"), 45.0),
    (re.compile(r"\b(forty|vierzig|quarante|cuarenta|[čc]ty[řr]icet)\b", re.I), 40.0),
    (re.compile(r"(四十)"), 40.0),
    (re.compile(r"\b(thirty[-\s]?five|fünfunddreissig|fünfunddreißig|trente-cinq|treinta\s+y\s+cinco|t[řr]icet\s+p[ěe]t)\b", re.I), 35.0),
    (re.compile(r"(三十五)"), 35.0),
    (re.compile(r"\b(thirty|dreissig|dreißig|trente|treinta|t[řr]icet)\b", re.I), 30.0),
    (re.compile(r"(三十)"), 30.0),
    (re.compile(r"\b(twenty[-\s]?five|fünfundzwanzig|vingt-cinq|veinticinco|dvacet\s+p[ěe]t)\b", re.I), 25.0),
    (re.compile(r"(二十五)"), 25.0),
    (re.compile(r"\b(twenty|zwanzig|zwänzg|vingt|veinte|dvacet)\b", re.I), 20.0),
    (re.compile(r"(二十)"), 20.0),
    (re.compile(r"\b(nineteen|neunzehn|dix-neuf|diecinueve|devaten[áa]ct)\b", re.I), 19.0),
    (re.compile(r"\b(eighteen|achtzehn|dix-huit|dieciocho|osmn[áa]ct)\b", re.I), 18.0),
    (re.compile(r"\b(seventeen|siebzehn|dix-sept|diecisiete|sedmn[áa]ct)\b", re.I), 17.0),
    (re.compile(r"\b(sixteen|sechzehn|seize|diecis[ée]is|[šs]estn[áa]ct)\b", re.I), 16.0),
    (re.compile(r"\b(fifteen|fünfzehn|fuenfzehn|quinze|quince|patn[áa]ct)\b", re.I), 15.0),
    (re.compile(r"\b(fourteen|vierzehn|quatorze|catorce|[čc]trn[áa]ct)\b", re.I), 14.0),
    (re.compile(r"\b(thirteen|dreizehn|treize|trece|t[řr]in[áa]ct)\b", re.I), 13.0),
    (re.compile(r"\b(twelve|zwölf|zwoelf|douze|doce|dvan[áa]ct)\b", re.I), 12.0),
    (re.compile(r"\b(eleven|elf|onze|once|jeden[áa]ct)\b", re.I), 11.0),
    (re.compile(r"\b(ten|zehn|zäh|dix|diez|deset)\b", re.I), 10.0),
    (re.compile(r"(十)"), 10.0),
    (re.compile(r"\b(nine|neun|nüün|neuf|nueve|dev[ěe]t)\b", re.I), 9.0),
    (re.compile(r"\b(eight|acht|huit|ocho|osm)\b", re.I), 8.0),
    (re.compile(r"\b(seven|sieben|sibe|sept|siete|sedm)\b", re.I), 7.0),
    (re.compile(r"\b(six|sechs|sächs|seis|[šs]est)\b", re.I), 6.0),
    (re.compile(r"\b(five|fünf|fuenf|füf|füüf|cinq|cinco|p[ěe]t)\b", re.I), 5.0),
    (re.compile(r"\b(four|vier|quatre|cuatro|[čc]ty[řr]i)\b", re.I), 4.0),
    (re.compile(r"\b(three|drei|drü|drüü|trois|tres|t[řr]i)\b", re.I), 3.0),
    (re.compile(r"\b(two|zwei|zwöi|deux|dos|dva|dv[ěe])\b", re.I), 2.0),
    (re.compile(r"\b(quarter|viertel|quart\b|cuarto|čtvrt|ctvrt)\b", re.I), 0.25),
    (re.compile(r"\b(half|halb|halbe|halben|demi|medio|media(?!\s+vuelta)|p[ůu]l)(?!-tour)\b", re.I), 0.5),
    (re.compile(r"\b(one|ein|eins|einen|äi|un|une|uno|una|jeden|jednu|jedna)\b", re.I), 1.0),
)

_CJK_NUM_MAP: dict[str, float] = {
    "零": 0.0,
    "〇": 0.0,
    "半": 0.5,
    "一": 1.0,
    "二": 2.0,
    "两": 2.0,
    "兩": 2.0,
    "三": 3.0,
    "四": 4.0,
    "五": 5.0,
    "六": 6.0,
    "七": 7.0,
    "八": 8.0,
    "九": 9.0,
    "十": 10.0,
}

_DIGIT_RE = re.compile(r"(?<![a-zA-Z])(\d+(?:[.,]\d+)?)")
_MM_UNIT_RE = re.compile(
    r"(?:\b|\d)\s*(?:mm|millimeters?|millimetres?|millimeter|millim[èe]tres?|mil[íi]metros?|milimetr\w*|毫米)\b",
    re.I,
)
_CM_UNIT_RE = re.compile(
    r"(?:\b|\d)\s*(?:cm|centimeters?|centimetres?|zentimeter|centim[èe]tres?|cent[íi]metros?|centimetr\w*|厘米|公分)\b",
    re.I,
)
_FT_UNIT_RE = re.compile(r"(?:\b|\d)\s*(?:ft|feet|foot)\b", re.I)
_INCH_UNIT_RE = re.compile(r"(?:\b|\d)\s*(?:inch|inches)\b", re.I)
_RAD_UNIT_RE = re.compile(r"(?:\b|\d)\s*(?:rad|radians?|radiant\w*|radianes|radi[áa]n\w*)\b|弧度", re.I)
_CCW_RE = re.compile(r"\b(counter-?clockwise|anti-?clockwise|gegen\s+den\s+uhrzeigersinn)\b", re.I)
_CW_RE = re.compile(r"\b(clockwise|im\s+uhrzeigersinn)\b", re.I)


def _lookup_spoken_int(token: str) -> int | None:
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    if t in {"a", "an"}:
        return 1
    if t in _SPOKEN_DIGIT_MAP:
        return _SPOKEN_DIGIT_MAP[t]
    if t.startswith("ein"):
        return 1
    return None


def _extract_raw_number(text: str) -> float | None:
    """Extract an unscaled numeric magnitude from digits, spoken decimals, compound halves, or words."""
    # 1. CJK compound halves (e.g. "一米半" -> 1.5, "两米半" -> 2.5) and CJK decimals ("二点五" -> 2.5)
    m_cjk_half = re.search(r"([一二两兩三四五六七八九十])\s*(?:米|公尺)半", text)
    if m_cjk_half:
        return _CJK_NUM_MAP[m_cjk_half.group(1)] + 0.5

    m_cjk_dec = re.search(r"([零〇一二两兩三四五六七八九十\d]+)[点點]([零〇一二三四五六七八九\d]+)", text)
    if m_cjk_dec:
        w_str, f_str = m_cjk_dec.group(1), m_cjk_dec.group(2)
        w_val = float(w_str) if w_str.isdigit() else sum(_CJK_NUM_MAP.get(ch, 0.0) for ch in w_str)
        if f_str.isdigit():
            f_digits = f_str
        else:
            f_digits = "".join(str(int(_CJK_NUM_MAP.get(ch, 0.0))) for ch in f_str)
        return float(f"{int(w_val)}.{f_digits}")

    # 2. German compound halves ("anderthalb", "zweieinhalb", etc.)
    for pattern, val in _GERMAN_COMPOUND_HALVES:
        if pattern.search(text):
            return val

    # 3. Spoken "X and a half" / "X et demi" / "X y medio" / "X a půl"
    m_half = _SPOKEN_AND_HALF_RE.search(text)
    if m_half:
        base_int = _lookup_spoken_int(m_half.group("base"))
        if base_int is not None:
            return float(base_int) + 0.5

    # 4. Spoken decimals ("two point five", "zwei komma fünf", "deux virgule cinq", "dos punto cinco")
    m_dec = _SPOKEN_DECIMAL_RE.search(text)
    if m_dec:
        whole_int = _lookup_spoken_int(m_dec.group("whole"))
        frac_tok = m_dec.group("frac").strip().lower().replace("-", " ")
        frac_str: str | None = None
        if frac_tok.isdigit():
            frac_str = frac_tok
        elif frac_tok == "twenty five":
            frac_str = "25"
        elif frac_tok == "seventy five":
            frac_str = "75"
        else:
            f_int = _lookup_spoken_int(frac_tok)
            if f_int is not None:
                frac_str = str(f_int)
        if whole_int is not None and frac_str is not None:
            return float(f"{whole_int}.{frac_str}")

    # 5. Standard decimal/integer digits ("5", "3.25", "0,5", "90")
    m_digit = _DIGIT_RE.search(text)
    if m_digit:
        return float(m_digit.group(1).replace(",", "."))

    # 6. Chinese tens + units ("二十五", "三十", "十一..十九")
    m_cjk_tens = re.search(r"([二三四五六七八九])十([一二三四五六七八九])?", text)
    if m_cjk_tens:
        tens = _CJK_NUM_MAP[m_cjk_tens.group(1)] * 10.0
        ones = _CJK_NUM_MAP[m_cjk_tens.group(2)] if m_cjk_tens.group(2) else 0.0
        return tens + ones

    m_teen = re.search(r"十([一二三四五六七八九])", text)
    if m_teen:
        return 10.0 + _CJK_NUM_MAP[m_teen.group(1)]

    # 7. Multilingual word numbers (ordered largest/longest first)
    for pattern, val in _WORD_NUMBERS:
        if pattern.search(text):
            return val

    # 8. Single CJK numerals ("五米", "半米", "九十度")
    m_cjk_unit = re.search(r"([半一二两兩三四五六七八九十])\s*(?:米|公尺|公分|厘米|毫米|度|弧度|步)", text)
    if m_cjk_unit:
        return _CJK_NUM_MAP[m_cjk_unit.group(1)]

    m_cjk = re.search(r"([半一二两兩三四五六七八九十])(?![下点點起直样樣定般])", text)
    if m_cjk:
        return _CJK_NUM_MAP[m_cjk.group(1)]

    return None


def _extract_distance_meters(text: str, default: float = 1.0) -> tuple[float, bool]:
    """Extract linear distance converted precisely to meters, plus `has_explicit_number`."""
    raw = _extract_raw_number(text)
    if raw is None:
        return default, False
    if _MM_UNIT_RE.search(text) or "毫米" in text:
        return round(raw / 1000.0, 4), True
    if _CM_UNIT_RE.search(text) or "厘米" in text or "公分" in text:
        return round(raw / 100.0, 4), True
    if _FT_UNIT_RE.search(text):
        return round(raw * 0.3048, 4), True
    if _INCH_UNIT_RE.search(text):
        return round(raw * 0.0254, 4), True
    return round(raw, 4), True


def _extract_numeric_value(text: str, default: float) -> tuple[float, bool]:
    """Backward-compatible helper returning `(distance_or_magnitude, has_explicit_number)`."""
    return _extract_distance_meters(text, default=default)


def _is_turn_around_phrase(lower: str) -> bool:
    return (
        any(
            kw in lower
            for kw in (
                "around",
                "demi-tour",
                "date la vuelta",
                "media vuelta",
                "180 grados",
                "dokola",
                "čelem vzad",
                "celem vzad",
                "180 stup",
                "um 180",
                "180 grad",
                "180°",
                "ume",
                "转身",
                "轉身",
                "掉头",
                "掉頭",
            )
        )
        or bool(
            re.search(
                r"\b(umdreh\w*|umdräi\w*|kehrt\w*)\b"
                r"|\b(dreh\w*|träi\w*|dräi\w*)\s+(?:bitte\s+)?(?:dich\s+|sie\s+sich\s+|sich\s+|euch\s+)?um\b(?!\s*\d)"
                r"|\boto[čc](?:te)?\s+se\b(?!.*?\b(?:doleva|vlevo|doprava|vpravo)\b)(?!\s+(?:o\s+)?\d)",
                lower,
            )
        )
    )


def _is_right_rotation(lower: str) -> bool:
    if _CCW_RE.search(lower):
        return False
    if _CW_RE.search(lower):
        return True
    return any(
        kw in lower
        for kw in ("right", "rechts", "rächts", "droite", "derecha", "doprava", "vpravo", "右")
    )


def _extract_angle_radians(text: str, *, is_turn_around: bool, is_right: bool) -> tuple[float, bool]:
    """Extract signed yaw angle in [-pi, +pi] radians (+CCW/left, -CW/right) and `has_explicit_angle`."""
    sign = -1.0 if is_right else 1.0
    raw = _extract_raw_number(text)
    if raw is None:
        deg = 180.0 if is_turn_around else 90.0
        yaw_rad = max(-3.14159, min(3.14159, round(sign * math.radians(deg), 5)))
        return yaw_rad, is_turn_around

    if _RAD_UNIT_RE.search(text):
        rad_mag = min(abs(raw), math.pi)
        yaw_rad = max(-3.14159, min(3.14159, round(sign * rad_mag, 5)))
        return yaw_rad, True

    deg = 180.0 if is_turn_around else min(abs(raw), 180.0)
    yaw_rad = max(-3.14159, min(3.14159, round(sign * math.radians(deg), 5)))
    return yaw_rad, True


@dataclass(frozen=True)
class ExplicitClauseSpec:
    """Exact kinematic parameters extracted deterministically from an operator instruction clause."""

    clause: str
    action: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw_rad: float | None = None
    speed: float = 0.25
    has_explicit_number: bool = False
    rationale: str = ""


def parse_explicit_clause_spec(text: str, language: str = "auto") -> ExplicitClauseSpec | None:
    """Parse a single clause into an `ExplicitClauseSpec` while leaving visual-only clauses as `None`."""
    cleaned = text.strip()
    if not cleaned or _QUESTION_GUARD_RE.search(cleaned) or _LOOK_FORWARD_IDIOM_RE.search(cleaned):
        return None

    if _ESTOP_RE.search(cleaned) or _STOP_CMD_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="stop",
            speed=0.0,
            has_explicit_number=True,
            rationale=f"Operator stop command ({language}): {cleaned[:120]}",
        )

    if _RETURN_HOME_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="return_home",
            x=0.0,
            y=0.0,
            z=0.0,
            speed=0.25,
            has_explicit_number=True,
            rationale=f"Return home command ({language}): {cleaned[:120]}",
        )

    m_lift = _TORSO_LIFT_RE.search(cleaned)
    if m_lift:
        verb = m_lift.group(1).lower()
        default_z = 0.0 if verb == "lower" else 0.30
        raw_z, has_num = _extract_distance_meters(cleaned, default=default_z)
        z_m = round(min(max(raw_z, 0.0), 0.60), 4)
        return ExplicitClauseSpec(
            clause=cleaned,
            action="move_to",
            x=0.0,
            y=0.0,
            z=z_m,
            speed=0.15,
            has_explicit_number=has_num or verb == "lower",
            rationale=f"Set Panther torso lift z={z_m:g}m ({language}): {cleaned[:120]}",
        )

    if _ROTATE_RE.search(cleaned):
        lower = cleaned.lower()
        is_turn_around = _is_turn_around_phrase(lower)
        is_right = _is_right_rotation(lower)
        yaw_rad, has_angle = _extract_angle_radians(
            cleaned,
            is_turn_around=is_turn_around,
            is_right=is_right,
        )
        if should_trigger_robot_camera(cleaned) and not has_angle:
            return None
        return ExplicitClauseSpec(
            clause=cleaned,
            action="rotate",
            yaw_rad=yaw_rad,
            speed=0.2,
            has_explicit_number=has_angle,
            rationale=f"Rotate command ({language}): {cleaned[:120]}",
        )

    raw_dist_m, has_dist = _extract_distance_meters(cleaned, default=1.0)
    dist_m = round(max(0.01, abs(raw_dist_m)), 4)

    # Do not override visual object-approach clauses that lack an explicit metric distance
    if should_trigger_robot_camera(cleaned) and not has_dist:
        return None

    if _FORWARD_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="move_to",
            x=dist_m,
            y=0.0,
            z=0.0,
            speed=0.25,
            has_explicit_number=has_dist,
            rationale=f"Move forward {dist_m:g}m ({language}): {cleaned[:120]}",
        )

    if _BACKWARD_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="move_to",
            x=-dist_m,
            y=0.0,
            z=0.0,
            speed=0.2,
            has_explicit_number=has_dist,
            rationale=f"Move backward {dist_m:g}m ({language}): {cleaned[:120]}",
        )

    if _STRAFE_LEFT_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="move_to",
            x=0.0,
            y=dist_m,
            z=0.0,
            speed=0.2,
            has_explicit_number=has_dist,
            rationale=f"Move left {dist_m:g}m ({language}): {cleaned[:120]}",
        )

    if _STRAFE_RIGHT_RE.search(cleaned):
        return ExplicitClauseSpec(
            clause=cleaned,
            action="move_to",
            x=0.0,
            y=-dist_m,
            z=0.0,
            speed=0.2,
            has_explicit_number=has_dist,
            rationale=f"Move right {dist_m:g}m ({language}): {cleaned[:120]}",
        )

    return None


def parse_multilingual_robot_intent(text: str, language: str = "auto") -> tuple[str, dict[str, Any]] | None:
    """Parse unambiguous robot motion/stop commands into (tool_name, arguments) in < 1ms."""
    cleaned = text.strip()
    if not cleaned or _QUESTION_GUARD_RE.search(cleaned) or _LOOK_FORWARD_IDIOM_RE.search(cleaned):
        return None

    # 1. Immediate Emergency Stop
    if _ESTOP_RE.search(cleaned):
        return "emergency_stop_robot", {"reason": cleaned[:200]}

    # 2. Controlled Stop
    if _STOP_CMD_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "stop",
            "speed": 0.0,
            "rationale": f"Operator stop command ({language}): {cleaned[:120]}",
        }

    # 3. Return Home
    if _RETURN_HOME_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "return_home",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "speed": 0.25,
            "rationale": f"Return home command ({language}): {cleaned[:120]}",
        }

    # 3b. Torso vertical lift
    m_lift = _TORSO_LIFT_RE.search(cleaned)
    if m_lift:
        verb = m_lift.group(1).lower()
        default_z = 0.0 if verb == "lower" else 0.30
        raw_z, _ = _extract_distance_meters(cleaned, default=default_z)
        z_m = round(min(max(raw_z, 0.0), 0.60), 4)
        return "propose_robot_action", {
            "action": "move_to",
            "x": 0.0,
            "y": 0.0,
            "z": z_m,
            "speed": 0.15,
            "rationale": f"Set Panther torso lift z={z_m:g}m ({language}): {cleaned[:120]}",
        }

    # 4. Rotation (check before left/right translation so "turn left" maps to rotate, not strafe)
    if _ROTATE_RE.search(cleaned):
        lower = cleaned.lower()
        is_turn_around = _is_turn_around_phrase(lower)
        is_right = _is_right_rotation(lower)
        yaw_rad, _ = _extract_angle_radians(
            cleaned,
            is_turn_around=is_turn_around,
            is_right=is_right,
        )
        return "propose_robot_action", {
            "action": "rotate",
            "yaw_rad": yaw_rad,
            "speed": 0.2,
            "rationale": f"Rotate command ({language}): {cleaned[:120]}",
        }

    # 5. Linear Translation: Forward (+x), Backward (-x), Left (+y), Right (-y)
    raw_dist_m, _ = _extract_distance_meters(cleaned, default=1.0)
    dist_m = round(max(0.01, abs(raw_dist_m)), 4)

    if _FORWARD_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "move_to",
            "x": dist_m,
            "y": 0.0,
            "z": 0.0,
            "speed": 0.25,
            "rationale": f"Move forward {dist_m:g}m ({language}): {cleaned[:120]}",
        }

    if _BACKWARD_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "move_to",
            "x": -dist_m,
            "y": 0.0,
            "z": 0.0,
            "speed": 0.2,
            "rationale": f"Move backward {dist_m:g}m ({language}): {cleaned[:120]}",
        }

    if _STRAFE_LEFT_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "move_to",
            "x": 0.0,
            "y": dist_m,
            "z": 0.0,
            "speed": 0.2,
            "rationale": f"Move left {dist_m:g}m ({language}): {cleaned[:120]}",
        }

    if _STRAFE_RIGHT_RE.search(cleaned):
        return "propose_robot_action", {
            "action": "move_to",
            "x": 0.0,
            "y": -dist_m,
            "z": 0.0,
            "speed": 0.2,
            "rationale": f"Move right {dist_m:g}m ({language}): {cleaned[:120]}",
        }

    return None


_FALLBACK_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and\s+then|then|and|puis|et\s+ensuite|et|und\s+dann|dann|und|dänn|und\s+dänn"
    r"|y\s+luego|y\s+despu[ée]s|luego|y(?!\s+(?:cinco|uno|una|dos|tres|cuatro|seis|siete|ocho|nueve)\b)"
    r"|a\s+pak|a\s+potom|pak|potom|a(?=\s+(?:oto[čc]|zato[čc]|jdi\b|jd[ěe]te\b|je[ďd]|popoje[ďd]|posu[ňn]|zastav|st[ůu]j|vra[ťt]|couvej|couvni)))\s+"
    r"|[，,;；]|然后|然後|接着|接著|再",
    re.IGNORECASE,
)


def detect_unhandled_robot_command(text: str, language: str = "auto") -> str | None:
    """Return ('emergency_stop_robot' | 'plan_robotics_task_with_er2' | None) if LLM omitted a mandatory robot tool call."""
    cleaned = text.strip()
    if not cleaned or _QUESTION_GUARD_RE.search(cleaned) or _LOOK_FORWARD_IDIOM_RE.search(cleaned):
        return None
    if _ESTOP_RE.search(cleaned):
        return "emergency_stop_robot"

    direct = parse_multilingual_robot_intent(cleaned, language)
    if direct is not None:
        return "emergency_stop_robot" if direct[0] == "emergency_stop_robot" else "plan_robotics_task_with_er2"

    for clause in _FALLBACK_CLAUSE_SPLIT_RE.split(cleaned):
        sub = clause.strip()
        if sub and parse_multilingual_robot_intent(sub, language) is not None:
            return "plan_robotics_task_with_er2"

    if _ROBOT_VISION_FALLBACK_RE.search(cleaned):
        return "plan_robotics_task_with_er2"

    return None


def format_robot_tool_reply(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    language: str,
    dialect: str | None = None,
) -> str | None:
    """Format a crisp 1-sentence spoken confirmation in the user's active language without a 2nd LLM hop."""
    is_zh_tw = language == "zh-TW" or any(t in language.lower() for t in ("tw", "hk", "hant"))
    is_zh = is_zh_tw or language.startswith(("zh", "cmn"))
    is_es = language.startswith("es")
    is_cs = language.startswith(("cs", "cz"))
    is_fr = language.startswith("fr")
    is_en = language.startswith("en")
    is_gsw = language == "gsw"

    if tool_name == "emergency_stop_robot":
        if is_zh_tw:
            return "緊急停止已啟動！所有機器人動作已立即中止。"
        if is_zh:
            return "紧急停止已激活！所有机器人动作已立即中止。"
        if is_es:
            return "¡Parada de emergencia activada! Todos los movimientos del robot se han detenido."
        if is_cs:
            return "Nouzové zastavení aktivováno! Všechny pohyby robota byly okamžitě zastaveny."
        if is_fr:
            return "Arrêt d'urgence activé ! Tous les mouvements du robot sont interrompus."
        if is_en:
            return "Emergency stop activated! All robot motion and pending commands are halted."
        if is_gsw:
            return "Not-Uus isch aktiviert! Alli Roboterbewegige sind sofort gstoppt worde."
        return "Not-Aus aktiviert! Alle Roboterbewegungen wurden sofort gestoppt."

    if tool_name == "get_robot_status":
        state = str(result.get("state", "ready"))
        batt = float(result.get("battery_percent", 95.0))
        if is_zh_tw:
            return f"機器人目前狀態為 {state}，剩餘電量 {batt:.0f}%。"
        if is_zh:
            return f"机器人当前状态为 {state}，剩余电量 {batt:.0f}%。"
        if is_es:
            return f"El robot está en estado {state} con {batt:.0f} por ciento de batería."
        if is_cs:
            return f"Robot je ve stavu {state} s {batt:.0f} procenty baterie."
        if is_fr:
            return f"Le robot est à l'état {state} avec {batt:.0f} pour cent de batterie."
        if is_en:
            return f"The robot is currently {state} with {batt:.0f} percent battery."
        if is_gsw:
            return f"De Roboter isch im Zustand {state} und d Batterie isch bi {batt:.0f} Prozänt."
        return f"Der Roboter ist im Status {state}, Akkustand {batt:.0f} Prozent."

    if tool_name == "get_robot_capabilities":
        max_r = result.get("max_workspace_radius_m", 10.0)
        max_v = result.get("max_linear_speed_mps", 0.5)
        if is_zh_tw:
            return f"機器人支援移動、旋轉、觀察與急停，安全半徑 {max_r:g} 公尺，最高速度每秒 {max_v:g} 公尺。"
        if is_zh:
            return f"机器人支持移动、旋转、观察与急停，安全半径 {max_r:g} 米，最高速度每秒 {max_v:g} 米。"
        if is_es:
            return f"El robot admite movimiento seguro hasta {max_r:g} metros a un máximo de {max_v:g} metros por segundo con confirmación humana."
        if is_cs:
            return f"Robot podporuje bezpečný pohyb až do {max_r:g} metrů rychlostí nejvýše {max_v:g} metru za sekundu s potvrzením operátora."
        if is_fr:
            return f"Le robot peut se déplacer jusqu'à {max_r:g} mètres à {max_v:g} mètre par seconde maximum avec confirmation humaine."
        if is_en:
            return f"The robot supports safe motion up to {max_r:g} meters at up to {max_v:g} meters per second with human confirmation."
        if is_gsw:
            return f"De Roboter cha sich sicher bis {max_r:g} Meter mit maximal {max_v:g} Meter pro Sekunde bewege."
        return f"Der Roboter unterstützt sichere Bewegungen bis {max_r:g} Meter bei maximal {max_v:g} Metern pro Sekunde."

    if tool_name == "switch_voice_dialect":
        label = result.get("label", dialect or "Wallisertiitsch")
        if is_gsw:
            return f"Alles klar, ich rede jetzt uf {label} wiiter."
        if is_es:
            return f"Se cambió el dialecto suizo alemán a {label}."
        if is_cs:
            return f"Švýcarský německý dialekt byl přepnut na {label}."
        if is_en:
            return f"Switched Swiss German voice dialect to {label}."
        if is_fr:
            return f"Le dialecte suisse allemand a été changé pour {label}."
        if is_zh:
            return f"已将瑞士德语方言切换为 {label}。"
        return f"Der Schweizerdeutsche Dialekt wurde auf {label} umgestellt."

    if tool_name != "propose_robot_action":
        return None

    if not result.get("validated", False):
        reason = str(result.get("rejected_reason", "Safety validation failed"))
        if is_zh_tw:
            return f"安全檢查攔截了該機器人指令：{reason}"
        if is_zh:
            return f"安全检查拦截了该机器人指令：{reason}"
        if is_es:
            return f"El control de seguridad bloqueó esa orden robótica: {reason}"
        if is_cs:
            return f"Bezpečnostní kontrola tento příkaz zablokovala: {reason}"
        if is_fr:
            return f"Le contrôle de sécurité a bloqué cette commande robotique : {reason}"
        if is_en:
            return f"Safety validation blocked that robot command: {reason}"
        if is_gsw:
            return f"D Sicherheitsprüefig hät de Roboterbefehl blockiert: {reason}"
        return f"Die Sicherheitsprüfung hat den Roboterbefehl blockiert: {reason}"

    cmd = result.get("command") or {}
    action = str(cmd.get("action") or arguments.get("action", "move_to"))
    target = cmd.get("target") or {}
    x = float(target.get("x", arguments.get("x", 0.0) or 0.0))
    y = float(target.get("y", arguments.get("y", 0.0) or 0.0))
    yaw = cmd.get("yaw_rad", arguments.get("yaw_rad"))
    obj_label = cmd.get("object_label") or arguments.get("object_label")

    if action == "stop":
        if is_zh_tw:
            return "已立即向機器人發送停止指令。"
        if is_zh:
            return "已立即向机器人发送停止指令。"
        if is_es:
            return "Orden de detención inmediata enviada al robot."
        if is_cs:
            return "Příkaz k okamžitému zastavení byl odeslán robotovi."
        if is_fr:
            return "Commande d'arrêt immédiat envoyée au robot."
        if is_en:
            return "Stopping the robot immediately."
        if is_gsw:
            return "De Roboter wird sofort gstoppt."
        return "Der Roboter wird sofort gestoppt."

    if action == "return_home":
        if is_zh_tw:
            return "已產生返回起點的動作建議，請在右側機器人面板中點擊確認。"
        if is_zh:
            return "已生成返回起点的动作建议，请在右侧机器人面板中点击确认。"
        if is_es:
            return "Propuesta de regreso a la base lista. Por favor confirme en el panel robótico."
        if is_cs:
            return "Návrh na návrat na základnu je připraven. Potvrďte jej prosím v robotickém panelu."
        if is_fr:
            return "Proposition de retour à la base prête. Veuillez confirmer dans le panneau robotique."
        if is_en:
            return "Ready to return to base. Please confirm the command in the robotics panel."
        if is_gsw:
            return "De Vorschlag zum zrugg zur Basis fahre isch parat. Bitte bestätig im Robotik-Panel."
        return "Vorschlag zur Rückkehr zur Basis ist bereit. Bitte bestätigen Sie im Robotik-Panel."

    if action == "rotate" and yaw is not None:
        deg_num = round(abs(math.degrees(float(yaw))))
        go_left = float(yaw) >= 0
        if is_zh_tw:
            return f"已產生向{'左' if go_left else '右'}旋轉 {deg_num} 度的動作建議，請在右側面板點擊確認。"
        if is_zh:
            return f"已生成向{'左' if go_left else '右'}旋转 {deg_num} 度的动作建议，请在右侧面板点击确认。"
        if is_es:
            return f"Propuesta para girar {deg_num} {_es_degrees(deg_num)} a la {'izquierda' if go_left else 'derecha'} lista. Por favor confirme en el panel robótico."
        if is_cs:
            return f"Návrh na otočení o {deg_num} {_cs_degrees(deg_num)} {'doleva' if go_left else 'doprava'} je připraven. Potvrďte jej prosím v robotickém panelu."
        if is_fr:
            return f"Proposition de rotation de {deg_num} {_fr_degrees(deg_num)} vers la {'gauche' if go_left else 'droite'} prête. Veuillez confirmer dans le panneau robotique."
        if is_en:
            return f"Ready to rotate {'left' if go_left else 'right'} {deg_num} {_en_degrees(deg_num)}. Please confirm in the robotics panel."
        if is_gsw:
            return f"De Vorschlag zum {deg_num} Grad nach {'links' if go_left else 'rächts'} träie isch parat. Bitte bestätig im Robotik-Panel."
        return f"Vorschlag für {deg_num} Grad Drehung nach {'links' if go_left else 'rechts'} ist bereit. Bitte bestätigen Sie im Robotik-Panel."

    if action == "move_to":
        if abs(y) < 1e-3 and x > 0:
            if is_zh_tw:
                return f"已產生向前移動 {x:g} 公尺的動作建議，請在右側機器人面板中點擊確認。"
            if is_zh:
                return f"已生成向前移动 {x:g} 米的动作建议，请在右侧机器人面板中点击确认。"
            if is_es:
                return f"Propuesta para avanzar {x:g} {_es_meters(x)} lista. Por favor confirme en el panel robótico."
            if is_cs:
                return f"Návrh na pohyb o {x:g} {_cs_meters(x)} dopředu je připraven. Potvrďte jej prosím v robotickém panelu."
            if is_fr:
                return f"Proposition d'avancer de {x:g} {_fr_meters(x)} prête. Veuillez confirmer dans le panneau robotique."
            if is_en:
                return f"Ready to move forward {x:g} {_en_meters(x)}. Please confirm the command in the robotics panel."
            if is_gsw:
                return f"De Vorschlag für {x:g} Meter vorwärts isch parat. Bitte bestätig s Kommando im Robotik-Panel."
            return f"Vorschlag für {x:g} Meter vorwärts ist bereit. Bitte bestätigen Sie im Robotik-Panel."

        if abs(y) < 1e-3 and x < 0:
            dx = abs(x)
            if is_zh_tw:
                return f"已產生向後退 {dx:g} 公尺的動作建議，請在右側機器人面板中點擊確認。"
            if is_zh:
                return f"已生成向后退 {dx:g} 米的动作建议，请在右侧机器人面板中点击确认。"
            if is_es:
                return f"Propuesta para retroceder {dx:g} {_es_meters(dx)} lista. Por favor confirme en el panel robótico."
            if is_cs:
                return f"Návrh na pohyb o {dx:g} {_cs_meters(dx)} dozadu je připraven. Potvrďte jej prosím v robotickém panelu."
            if is_fr:
                return f"Proposition de reculer de {dx:g} {_fr_meters(dx)} prête. Veuillez confirmer dans le panneau robotique."
            if is_en:
                return f"Ready to move backward {dx:g} {_en_meters(dx)}. Please confirm in the robotics panel."
            if is_gsw:
                return f"De Vorschlag für {dx:g} Meter zrugg isch parat. Bitte bestätig s Kommando im Robotik-Panel."
            return f"Vorschlag für {dx:g} Meter rückwärts ist bereit. Bitte bestätigen Sie im Robotik-Panel."

        if abs(x) < 1e-3 and abs(y) > 0:
            dy = abs(y)
            left = y > 0
            if is_zh_tw:
                return f"已產生向{'左' if left else '右'}移動 {dy:g} 公尺的動作建議，請在右側面板點擊確認。"
            if is_zh:
                return f"已生成向{'左' if left else '右'}移动 {dy:g} 米的动作建议，请在右侧面板点击确认。"
            if is_es:
                return f"Propuesta para desplazarse {dy:g} {_es_meters(dy)} a la {'izquierda' if left else 'derecha'} lista. Por favor confirme."
            if is_cs:
                return f"Návrh na posun o {dy:g} {_cs_meters(dy)} {'doleva' if left else 'doprava'} je připraven. Potvrďte jej prosím."
            if is_fr:
                return f"Proposition de déplacement de {dy:g} {_fr_meters(dy)} vers la {'gauche' if left else 'droite'} prête. Veuillez confirmer."
            if is_en:
                return f"Ready to move {'left' if left else 'right'} {dy:g} {_en_meters(dy)}. Please confirm in the robotics panel."
            if is_gsw:
                return f"De Vorschlag für {dy:g} Meter nach {'links' if left else 'rächts'} isch parat. Bitte bestätig im Robotik-Panel."
            return f"Vorschlag für {dy:g} Meter nach {'links' if left else 'rechts'} ist bereit. Bitte bestätigen Sie im Robotik-Panel."

        if is_zh_tw:
            return f"已產生移動至座標 ({x:g}, {y:g}) 公尺的動作建議，請在面板中確認。"
        if is_zh:
            return f"已生成移动至坐标 ({x:g}, {y:g}) 米的动作建议，请在面板中确认。"
        if is_es:
            return f"Propuesta para ir a las coordenadas ({x:g}, {y:g}) metros lista. Por favor confirme."
        if is_cs:
            return f"Návrh na přesun na souřadnice ({x:g}, {y:g}) metrů je připraven. Potvrďte jej prosím."
        if is_fr:
            return f"Proposition de déplacement vers ({x:g}, {y:g}) mètres prête. Veuillez confirmer."
        if is_en:
            return f"Ready to move to coordinates ({x:g}, {y:g}) meters. Please confirm in the robotics panel."
        if is_gsw:
            return f"De Vorschlag zur Position ({x:g}, {y:g}) Meter isch parat. Bitte bestätig im Robotik-Panel."
        return f"Vorschlag zur Position ({x:g}, {y:g}) Meter ist bereit. Bitte bestätigen Sie im Robotik-Panel."

    # Non-translation actions (inspect_object, look_at, pick_object, place_object)
    awaiting = bool(result.get("awaiting_human_confirmation", False))
    target_txt = f" ({obj_label})" if obj_label else ""
    if is_zh_tw:
        suffix = "，請在右側機器人面板點擊確認。" if awaiting else "，已通過安全檢查。"
        return f"機器人動作 {action}{target_txt} 已就緒{suffix}"
    if is_zh:
        suffix = "，请在右侧机器人面板点击确认。" if awaiting else "，已通过安全检查。"
        return f"机器人动作 {action}{target_txt} 已就绪{suffix}"
    if is_es:
        suffix = " Por favor confirme en el panel robótico." if awaiting else ""
        return f"La acción robótica {action}{target_txt} ha sido validada.{suffix}"
    if is_cs:
        suffix = " Potvrďte ji prosím v robotickém panelu." if awaiting else ""
        return f"Robotická akce {action}{target_txt} prošla bezpečnostní kontrolou.{suffix}"
    if is_fr:
        suffix = " Veuillez confirmer dans le panneau robotique." if awaiting else ""
        return f"L'action robotique {action}{target_txt} a été validée.{suffix}"
    if is_en:
        suffix = " Please confirm in the robotics panel." if awaiting else ""
        return f"Robot action {action}{target_txt} has been safety-checked.{suffix}"
    if is_gsw:
        suffix = " Bitte bestätig im Robotik-Panel." if awaiting else ""
        return f"D Roboter-Aktion {action}{target_txt} isch prüeft worde.{suffix}"
    suffix = " Bitte bestätigen Sie im Robotik-Panel." if awaiting else ""
    return f"Die Roboteraktion {action}{target_txt} wurde sicherheitsgeprüft.{suffix}"


def _en_meters(val: float) -> str:
    return "meter" if abs(abs(val) - 1.0) < 1e-6 else "meters"


def _fr_meters(val: float) -> str:
    return "mètre" if abs(abs(val) - 1.0) < 1e-6 else "mètres"


def _es_meters(val: float) -> str:
    return "metro" if abs(abs(val) - 1.0) < 1e-6 else "metros"


def _cs_meters(val: float) -> str:
    av = abs(val)
    if abs(av - round(av)) > 1e-6:
        return "metru"
    n = int(round(av))
    if n == 1:
        return "metr"
    if 2 <= n <= 4:
        return "metry"
    return "metrů"


def _en_degrees(deg: int) -> str:
    return "degree" if abs(deg) == 1 else "degrees"


def _fr_degrees(deg: int) -> str:
    return "degré" if abs(deg) == 1 else "degrés"


def _es_degrees(deg: int) -> str:
    return "grado" if abs(deg) == 1 else "grados"


def _cs_degrees(deg: int) -> str:
    ad = abs(deg)
    if ad == 1:
        return "stupeň"
    if 2 <= ad <= 4:
        return "stupně"
    return "stupňů"


def default_quick_ack(language: str, dialect: str | None = None) -> str:
    """Short spoken acknowledgment when Gemini Flash delegates heavy planning to ER 2."""
    is_zh_tw = language == "zh-TW" or any(t in language.lower() for t in ("tw", "hk", "hant"))
    if is_zh_tw:
        return "好的，沒問題！正在為您規劃動作。"
    if language.startswith(("zh", "cmn")):
        return "好的，没问题！正在为您规划动作。"
    if language.startswith("es"):
        return "¡Claro, enseguida! Estoy planificando esa maniobra."
    if language.startswith(("cs", "cz")):
        return "Jasně, hned to naplánuji!"
    if language.startswith("fr"):
        return "Bien sûr ! Je prépare cette manœuvre."
    if language.startswith("en"):
        return "Yeah sure, planning that now!"
    if language == "gsw":
        return "Ja sicher, ich plane das grad!"
    return "Ja klar, ich plane das sofort!"


def _describe_single_step(cmd: dict[str, Any], language: str) -> str:
    action = str(cmd.get("action", "move_to"))
    target = cmd.get("target") or {}
    x = float(target.get("x", 0.0) or 0.0)
    y = float(target.get("y", 0.0) or 0.0)
    z = float(target.get("z", 0.0) or 0.0)
    yaw = cmd.get("yaw_rad")
    obj = cmd.get("object_label")

    is_zh_tw = language == "zh-TW" or any(t in language.lower() for t in ("tw", "hk", "hant"))
    is_zh = is_zh_tw or language.startswith(("zh", "cmn"))
    is_es = language.startswith("es")
    is_cs = language.startswith(("cs", "cz"))
    is_fr = language.startswith("fr")
    is_en = language.startswith("en")
    is_gsw = language == "gsw"

    if action == "move_to":
        if abs(x) < 1e-3 and abs(y) < 1e-3 and z > 1e-3:
            if is_zh_tw:
                return f"將升降台調整至 {z:g} 公尺"
            if is_zh:
                return f"将升降台调整至 {z:g} 米"
            if is_es:
                return f"ajustar el elevador a {z:g} {_es_meters(z)}"
            if is_cs:
                return f"nastavit zdvih na {z:g} {_cs_meters(z)}"
            if is_fr:
                return f"régler l'élévateur à {z:g} {_fr_meters(z)}"
            if is_en:
                return f"set torso lift to {z:g} {_en_meters(z)}"
            if is_gsw:
                return f"de Lift uf {z:g} Meter stelle"
            return f"den Torso-Lift auf {z:g} Meter stellen"
        if abs(y) < 1e-3 and x > 0:
            if is_zh_tw:
                return f"向前移動 {x:g} 公尺"
            if is_zh:
                return f"向前移动 {x:g} 米"
            if is_es:
                return f"avanzar {x:g} {_es_meters(x)}"
            if is_cs:
                return f"jet {x:g} {_cs_meters(x)} dopředu"
            if is_fr:
                return f"avancer de {x:g} {_fr_meters(x)}"
            if is_en:
                return f"move forward {x:g} {_en_meters(x)}"
            if is_gsw:
                return f"{x:g} Meter vorwärts fahre"
            return f"{x:g} Meter vorwärts fahren"
        if abs(y) < 1e-3 and x < 0:
            dx = abs(x)
            if is_zh_tw:
                return f"向後退 {dx:g} 公尺"
            if is_zh:
                return f"向后退 {dx:g} 米"
            if is_es:
                return f"retroceder {dx:g} {_es_meters(dx)}"
            if is_cs:
                return f"couvnout o {dx:g} {_cs_meters(dx)}"
            if is_fr:
                return f"reculer de {dx:g} {_fr_meters(dx)}"
            if is_en:
                return f"move backward {dx:g} {_en_meters(dx)}"
            if is_gsw:
                return f"{dx:g} Meter zrugg fahre"
            return f"{dx:g} Meter rückwärts fahren"
        if abs(x) < 1e-3 and abs(y) > 0:
            dy = abs(y)
            left = y > 0
            if is_zh_tw:
                return f"向{'左' if left else '右'}移動 {dy:g} 公尺"
            if is_zh:
                return f"向{'左' if left else '右'}移动 {dy:g} 米"
            if is_es:
                return f"desplazarse {dy:g} {_es_meters(dy)} a la {'izquierda' if left else 'derecha'}"
            if is_cs:
                return f"posunout se o {dy:g} {_cs_meters(dy)} {'doleva' if left else 'doprava'}"
            if is_fr:
                return f"aller à {'gauche' if left else 'droite'} de {dy:g} {_fr_meters(dy)}"
            if is_en:
                return f"move {'left' if left else 'right'} {dy:g} {_en_meters(dy)}"
            if is_gsw:
                return f"{dy:g} Meter nach {'links' if left else 'rächts'} fahre"
            return f"{dy:g} Meter nach {'links' if left else 'rechts'} fahren"
        if is_zh_tw:
            return f"移動至 ({x:g}, {y:g}) 公尺"
        if is_zh:
            return f"移动至 ({x:g}, {y:g}) 米"
        if is_es:
            return f"ir a ({x:g}, {y:g}) metros"
        if is_cs:
            return f"přesunout se na ({x:g}, {y:g}) metrů"
        if is_fr:
            return f"aller vers ({x:g}, {y:g}) mètres"
        if is_en:
            return f"move to ({x:g}, {y:g}) meters"
        if is_gsw:
            return f" uf ({x:g}, {y:g}) Meter fahre".strip()
        return f"zu ({x:g}, {y:g}) Metern fahren"

    if action == "rotate" and yaw is not None:
        deg = round(abs(math.degrees(float(yaw))))
        if deg >= 175:
            if is_zh_tw:
                return "掉頭旋轉 180 度"
            if is_zh:
                return "掉头旋转 180 度"
            if is_es:
                return "darse la vuelta 180 grados"
            if is_cs:
                return "otočit se o 180 stupňů"
            if is_fr:
                return "faire demi-tour à 180 degrés"
            if is_en:
                return "turn around 180 degrees"
            if is_gsw:
                return "um 180 Grad umdräie"
            return "um 180 Grad umdrehen"
        left = float(yaw) >= 0
        if is_zh_tw:
            return f"向{'左' if left else '右'}旋轉 {deg} 度"
        if is_zh:
            return f"向{'左' if left else '右'}旋转 {deg} 度"
        if is_es:
            return f"girar {deg} {_es_degrees(deg)} a la {'izquierda' if left else 'derecha'}"
        if is_cs:
            return f"otočit se o {deg} {_cs_degrees(deg)} {'doleva' if left else 'doprava'}"
        if is_fr:
            return f"tourner à {'gauche' if left else 'droite'} de {deg} {_fr_degrees(deg)}"
        if is_en:
            return f"turn {'left' if left else 'right'} {deg} {_en_degrees(deg)}"
        if is_gsw:
            return f"{deg} Grad nach {'links' if left else 'rächts'} träie"
        return f"{deg} Grad nach {'links' if left else 'rechts'} drehen"

    if action == "return_home":
        if is_zh_tw:
            return "返回起點"
        if is_zh:
            return "返回起点"
        if is_es:
            return "volver a la base"
        if is_cs:
            return "vrátit se na základnu"
        if is_fr:
            return "retourner à la base"
        if is_en:
            return "return to base"
        if is_gsw:
            return "zrugg zur Basis fahre"
        return "zur Basis zurückkehren"

    if action == "stop":
        if is_zh_tw:
            return "立即停止"
        if is_zh:
            return "立即停止"
        if is_es:
            return "detenerse de inmediato"
        if is_cs:
            return "okamžitě zastavit"
        if is_fr:
            return "s'arrêter immédiatement"
        if is_en:
            return "stop immediately"
        if is_gsw:
            return "sofort stoppe"
        return "sofort anhalten"

    if action in {"inspect_object", "look_at"}:
        tgt_name = obj or "workspace"
        if is_zh_tw:
            return f"用相機檢查 {tgt_name}"
        if is_zh:
            return f"用相机检查 {tgt_name}"
        if is_es:
            return f"inspeccionar {tgt_name} con la cámara"
        if is_cs:
            return f"zkontrolovat {tgt_name} kamerou"
        if is_fr:
            return f"examiner {tgt_name} avec la caméra"
        if is_en:
            return f"inspect {tgt_name} with camera"
        if is_gsw:
            return f"{tgt_name} mit de Kamera aluege"
        return f"{tgt_name} mit der Kamera prüfen"

    if action == "pick_object":
        tgt_name = obj or "object"
        if is_zh_tw:
            return f"用夾爪抓取 {tgt_name}"
        if is_zh:
            return f"用夹爪抓取 {tgt_name}"
        if is_es:
            return f"sujetar {tgt_name} con la pinza"
        if is_cs:
            return f"uchopit {tgt_name} chapadlem"
        if is_fr:
            return f"saisir {tgt_name} avec la pince"
        if is_en:
            return f"grasp {tgt_name} with gripper"
        if is_gsw:
            return f"{tgt_name} mit em Griifer neh"
        return f"{tgt_name} mit dem Greifer aufnehmen"

    if action == "place_object":
        tgt_name = obj or "object"
        if is_zh_tw:
            return f"鬆開夾爪放下 {tgt_name}"
        if is_zh:
            return f"松开夹爪放下 {tgt_name}"
        if is_es:
            return f"soltar {tgt_name} abriendo la pinza"
        if is_cs:
            return f"položit {tgt_name} otevřením chapadla"
        if is_fr:
            return f"déposer {tgt_name} en ouvrant la pince"
        if is_en:
            return f"release {tgt_name} with gripper"
        if is_gsw:
            return f"{tgt_name} mit em Griifer ablege"
        return f"{tgt_name} mit dem Greifer ablegen"

    label_part = f" ({obj})" if obj else ""
    return f"{action}{label_part}"


def format_er2_plan_summary(
    validated_commands: list[dict[str, Any]],
    rejected_reasons: list[str],
    language: str,
    *,
    scene_summary: str | None = None,
    camera_captured: bool = False,
) -> str:
    """Format a localized spoken summary of a multi-step Gemini Robotics ER 2 plan."""
    is_zh_tw = language == "zh-TW" or any(t in language.lower() for t in ("tw", "hk", "hant"))
    is_zh = is_zh_tw or language.startswith(("zh", "cmn"))
    is_es = language.startswith("es")
    is_cs = language.startswith(("cs", "cz"))
    is_fr = language.startswith("fr")
    is_en = language.startswith("en")
    is_gsw = language == "gsw"

    cam_prefix = ""
    if camera_captured:
        if is_zh_tw:
            cam_prefix = "已拍攝機器人相機畫面。"
        elif is_zh:
            cam_prefix = "已拍摄机器人相机画面。"
        elif is_es:
            cam_prefix = "Cámara del robot capturada. "
        elif is_cs:
            cam_prefix = "Snímek z kamery robota pořízen. "
        elif is_fr:
            cam_prefix = "Image caméra du robot capturée. "
        elif is_en:
            cam_prefix = (
                f"{scene_summary.strip()} "
                if scene_summary and len(scene_summary.strip()) <= 110
                else "Robot camera frame analyzed. "
            )
        elif is_gsw:
            cam_prefix = "Kamerabild vom Roboter isch analysiert. "
        else:
            cam_prefix = "Kamerabild vom Roboter analysiert. "

    if not validated_commands:
        if camera_captured and scene_summary and not rejected_reasons:
            return f"{cam_prefix}{scene_summary}" if not is_en else scene_summary
        reason = rejected_reasons[0] if rejected_reasons else "No actionable steps produced"
        if is_zh_tw:
            return f"安全檢查攔截了該計畫：{reason}"
        if is_zh:
            return f"安全检查拦截了该计划：{reason}"
        if is_es:
            return f"El control de seguridad bloqueó ese plan: {reason}"
        if is_cs:
            return f"Bezpečnostní kontrola tento plán zablokovala: {reason}"
        if is_fr:
            return f"Le contrôle de sécurité a bloqué ce plan : {reason}"
        if is_en:
            return f"Safety validation blocked that plan: {reason}"
        if is_gsw:
            return f"D Sicherheitsprüefig hät de Plan blockiert: {reason}"
        return f"Die Sicherheitsprüfung hat diesen Plan blockiert: {reason}"

    steps = [_describe_single_step(c, language) for c in validated_commands]
    needs_confirm = any(c.get("requires_confirmation") and not c.get("confirmed_by_human") for c in validated_commands)

    if len(steps) == 1:
        s0 = steps[0]
        if is_zh_tw:
            tail = "，請在右側機器人面板點擊確認。" if needs_confirm else "。"
            return f"{cam_prefix}已規劃：{s0}{tail}"
        if is_zh:
            tail = "，请在右侧机器人面板点击确认。" if needs_confirm else "。"
            return f"{cam_prefix}已规划：{s0}{tail}"
        if is_es:
            tail = " Por favor confirme en el panel robótico." if needs_confirm else ""
            return f"{cam_prefix}Acción lista: {s0}.{tail}"
        if is_cs:
            tail = " Potvrďte prosím v robotickém panelu." if needs_confirm else ""
            return f"{cam_prefix}Připraveno: {s0}.{tail}"
        if is_fr:
            tail = " Veuillez confirmer dans le panneau robotique." if needs_confirm else ""
            return f"{cam_prefix}Action prête : {s0}.{tail}"
        if is_en:
            tail = " Please confirm in the robotics panel." if needs_confirm else ""
            return f"{cam_prefix}Queued: {s0}.{tail}"
        if is_gsw:
            tail = " Bitte bestätig im Robotik-Panel." if needs_confirm else ""
            return f"{cam_prefix}Parat: {s0}.{tail}"
        tail = " Bitte bestätigen Sie im Robotik-Panel." if needs_confirm else ""
        return f"{cam_prefix}Bereit: {s0}.{tail}"

    if is_zh_tw:
        joined = "，然後".join(steps)
        tail = "。請在右側機器人面板依序確認。" if needs_confirm else "。"
        return f"{cam_prefix}已規劃 {len(steps)} 個步驟：{joined}{tail}"
    if is_zh:
        joined = "，然后".join(steps)
        tail = "。请在右侧机器人面板依次确认。" if needs_confirm else "。"
        return f"{cam_prefix}已规划 {len(steps)} 个步骤：{joined}{tail}"
    if is_es:
        joined = ", luego ".join(steps)
        tail = " Por favor confirme en el panel robótico." if needs_confirm else ""
        return f"{cam_prefix}Plan de {len(steps)} pasos listo: {joined}.{tail}"
    if is_cs:
        joined = ", pak ".join(steps)
        tail = " Potvrďte prosím v robotickém panelu." if needs_confirm else ""
        return f"{cam_prefix}Plán o {len(steps)} krocích připraven: {joined}.{tail}"
    if is_fr:
        joined = ", puis ".join(steps)
        tail = " Veuillez confirmer dans le panneau robotique." if needs_confirm else ""
        return f"{cam_prefix}Plan en {len(steps)} étapes prêt : {joined}.{tail}"
    if is_en:
        joined = ", then ".join(steps)
        tail = " Please confirm in the robotics panel." if needs_confirm else ""
        return f"{cam_prefix}Queued {len(steps)} steps: {joined}.{tail}"
    if is_gsw:
        joined = ", dänn ".join(steps)
        tail = " Bitte bestätig im Robotik-Panel." if needs_confirm else ""
        return f"{cam_prefix}Plan mit {len(steps)} Schritt parat: {joined}.{tail}"
    joined = ", dann ".join(steps)
    tail = " Bitte bestätigen Sie im Robotik-Panel." if needs_confirm else ""
    return f"{cam_prefix}Plan mit {len(steps)} Schritten bereit: {joined}.{tail}"
