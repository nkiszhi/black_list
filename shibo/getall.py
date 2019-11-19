import csv
import time
import os
from deduplicate import deduplicate
from trails.feeds.chinad import fetch as chinadfetch
from trails.feeds.conficker import fetch as confickerfetch
from trails.feeds.cryptolocker import fetch as cryptolockerfetch
from trails.feeds.gameover import fetch as gameoverfetch
from trails.feeds.locky import fetch as lockyfetch
from trails.feeds.necurs import fetch as necursfetch
from trails.feeds.tofsee import fetch as tofseefetch
from trails.feeds.virut import fetch as virutfetch
from trails.feeds.abuseipdb import fetch as abuseipdbfetch
from trails.feeds.alienvault import fetch as alienvaultfetch
from trails.feeds.atmos import fetch as atmosfetch
from trails.feeds.badips import fetch as badipsfetch
from trails.feeds.bambenekconsultingc2dns import fetch as dnsfetch
from trails.feeds.bambenekconsultingdga import fetch as dgafetch
from trails.feeds.bitcoinnodes import fetch as bitfetch
from trails.feeds.blackbook import fetch as blackfetch
from trails.feeds.blocklist import fetch as blockfetch
from trails.feeds.botscout import fetch as botfetch
from trails.feeds.bruteforceblocker import fetch as brufetch
from trails.feeds.ciarmy import fetch as ciafetch
from trails.feeds.cobaltstrike import fetch as cobfetch
from trails.feeds.cruzit import fetch as crufetch
from trails.feeds.cybercrimetracker import fetch as cybfetch
from trails.feeds.dataplane import fetch as datafetch
from trails.feeds.dshielddns import fetch as nsfetch
from trails.feeds.dshieldip import fetch as dsfetch
from trails.feeds.emergingthreatsbot import fetch as emfetch
from trails.feeds.emergingthreatscip import fetch as cipfetch
from trails.feeds.emergingthreatsdns import fetch as emerfetch
from trails.feeds.greensnow import fetch as snowfetch
from trails.feeds.loki import fetch as lokifetch
from trails.feeds.magento import fetch as magfetch
from trails.feeds.malc0de import fetch as malfetch
from trails.feeds.malwaredomainlistdns import fetch as maldnsfetch
from trails.feeds.malwaredomains import fetch as malmainsfetch
from trails.feeds.maxmind import fetch as maxmindfetch
from trails.feeds.minerchk import fetch as minerfetch
from trails.feeds.myip import fetch as myipfetch
from trails.feeds.nothink import fetch as nothinkfetch
from trails.feeds.openphish import fetch as openfetch
from trails.feeds.policeman import fetch as policefetch
from trails.feeds.pony import fetch as ponyfetch
from trails.feeds.proxyrss import fetch as proxyrssfetch
from trails.feeds.proxylists import fetch as proxylistsfetch
from trails.feeds.proxyspy import fetch as proxyspyfetch
from trails.feeds.ransomwaretrackerdns import fetch as randnsfetch
from trails.feeds.ransomwaretrackerip import fetch as ranipfetch
from trails.feeds.ransomwaretrackerurl import fetch as ranurlfetch
from trails.feeds.riproxies import fetch as ripfetch
from trails.feeds.rutgers import fetch as rutfetch
from trails.feeds.sblam import fetch as sblamfetch
from trails.feeds.socksproxy import fetch as sockfetch
from trails.feeds.sslipbl import fetch as sslipfetch
from trails.feeds.sslproxies import fetch as sslprofetch
from trails.feeds.talosintelligence import fetch as talofetch
from trails.feeds.torproject import fetch as torfetch
from trails.feeds.torstatus import fetch as torsfetch
from trails.feeds.trickbot import fetch as trickbotfetch
from trails.feeds.turris import fetch as turrisfetch
from trails.feeds.urlhaus import fetch as urlhausfetch
from trails.feeds.urlvir import fetch as urlvirfetch
from trails.feeds.voipbl import fetch as voipblfetch
from trails.feeds.vxvault import fetch as vxfetch
a=[chinadfetch(),confickerfetch(),cryptolockerfetch(),gameoverfetch(),lockyfetch(),necursfetch(),tofseefetch(),virutfetch(),abuseipdbfetch(),alienvaultfetch(),atmosfetch(),badipsfetch(),dnsfetch(),dgafetch(),bitfetch(),blackfetch(),blockfetch(),botfetch(),brufetch(),ciafetch(),cobfetch(),crufetch(),cybfetch(),datafetch(),nsfetch(),dsfetch(),emfetch(),cipfetch(),emerfetch(),snowfetch(),lokifetch(),magfetch(),malfetch(),maldnsfetch(),malmainsfetch(),maxmindfetch(),minerfetch(),myipfetch(),nothinkfetch(),openfetch(),policefetch(),ponyfetch(),proxyrssfetch(),proxylistsfetch(),proxyspyfetch(),randnsfetch(),ranipfetch(),ranurlfetch(),ripfetch(),rutfetch(),sblamfetch(),sockfetch(),sslipfetch(),sslprofetch(),talofetch(),torfetch(),torsfetch(),trickbotfetch(),turrisfetch(),urlhausfetch(),urlvirfetch(),voipblfetch(),vxfetch()]
directory="/root/blacklist"
os.chdir(directory)
t = time.strftime('%Y-%m-%d',time.localtime())
suffix=".csv"
newfile=t+suffix
if not os.path.exists(newfile):
    f=open(newfile,'a')
    write=csv.writer(f)
    for i in a:
        re=i
        for k,v in re.items():
            x,y=v
            row=[k,x,y]
            write.writerow(row)
else:
    f=open(newfile,'a')
    write=csv.writer(f)
    for i in a:
        re=i
        for k,v in re.items():
            x,y=v
            row=[k,x,y]
            write.writerow(row)
    deduplicate(newfile)
