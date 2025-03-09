from ryu.base import app_manager
from ryu.controller import ofp_event
from collections import defaultdict
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase
from ryu.topology import event

# Switches
switches = []

# mymacs[srcmac]->(switch, port)
mymacs = {}

# adjacency map [sw1][sw2]->port from sw1 to sw2
adjacency = defaultdict(lambda: defaultdict(lambda: None))

class CustomController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CustomController, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.datapath_list = []

    # Get shortest path using Dijkstra's algorithm
    def get_shortest_path(self, src, dst, first_port, final_port):
        # Execute Dijkstra's algorithm

        # Define dictionaries to store each node's distance and its previous node in the path from the first node to that node
        distance = {}
        previous = {}

        # Set initial distance of every node to infinity
        for dpid in switches:
            distance[dpid] = float('inf')
            previous[dpid] = None

        # Set distance of the source to 0
        distance[src] = 0

        # Create a set of all nodes
        Q = set(switches)

        # Find shortest path
        while Q:
            # Get the closest node to src among undiscovered nodes
            u = min(Q, key=lambda x: distance[x])
            Q.remove(u)

            # Calculate minimum distance for all adjacent nodes to u
            for p in switches:
                # If u and other switches are adjacent
                if adjacency[u][p] is not None:
                    # Set the weight to 1 so that we count the number of routers in the path
                    w = 1
                    # If the path via u to p has lower cost, then update the cost
                    if distance[u] + w < distance[p]:
                        distance[p] = distance[u] + w
                        previous[p] = u

        # Create a list of switches between src and dst which are in the shortest path obtained by Dijkstra's algorithm reversely
        r = []
        p = dst
        r.append(p)
        # Set q to the last node before dst
        q = previous[p]
        while q is not None:
            if q == src:
                r.append(q)
                break
            p = q
            r.append(p)
            q = previous[p]

        # Reverse r as it was from dst to src
        r.reverse()

        # Set path
        if src == dst:
            path = [src]
        else:
            path = r

        # Now adding in_port and out_port to the path
        r = []
        in_port = first_port
        for s1, s2 in zip(path[:-1], path[1:]):
            out_port = adjacency[s1][s2]
            r.append((s1, in_port, out_port))
            in_port = adjacency[s2][s1]
        r.append((dst, in_port, final_port))
        return r

    # Define event handler for switch features setup and configuration
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        # Get the datapath, ofproto and parser objects of the event
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Set match condition to anything
        match = parser.OFPMatch()
        # Set action to send packets to OpenFlow Controller without buffering
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        # Set priority to 0 to match any packet inside any flow table
        mod = parser.OFPFlowMod(
                            datapath=datapath, match=match, cookie=0,
                            command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0,
                            priority=0, instructions=inst)
        # Finalize the mod
        datapath.send_msg(mod)

    # Define event handler for packets coming to switches
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        # Get msg, datapath, ofproto and parser objects
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Get the port switch received the packet with
        in_port = msg.match['in_port']
        # Create a packet encoder/decoder class with the raw data obtained by msg
        pkt = packet.Packet(msg.data)
        # Get the protocol that matches the received packet
        eth = pkt.get_protocol(ethernet.ethernet)

        # Avoid LLDP broadcasts
        if eth.ethertype == 35020 or eth.ethertype == 34525:
            return

        # Get source and destination of the link
        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        # Add the host to the mymacs of the first switch that gets the packet
        if src not in mymacs:
            mymacs[src] = (dpid, in_port)

        # Find shortest path if destination exists in mymacs
        if dst in mymacs:
            p = self.get_shortest_path(mymacs[src][0], mymacs[dst][0], mymacs[src][1], mymacs[dst][1])
            self.install_path(p, ev, src, dst)
            out_port = p[0][2]
        else:
            out_port = ofproto.OFPP_FLOOD

        # Get actions part of the flow table
        actions = [parser.OFPActionOutput(out_port)]

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
                                  actions=actions, data=data)
        datapath.send_msg(out)

    # Define event handler for adding/deleting of switches, hosts, ports and links event
    events = [event.EventSwitchEnter, event.EventSwitchLeave, event.EventPortAdd,
              event.EventPortDelete, event.EventPortModify, event.EventLinkAdd, event.EventLinkDelete]

    @set_ev_cls(events)
    def update_topology_data(self, ev):
        global switches
        # Get the list of switches from the topology API
        switch_list = get_switch(self.topology_api_app, None)
        # Create a list of switch dpid
        switches = [switch.dp.id for switch in switch_list]
        # Get the list of links from the topology API
        links_list = get_link(self.topology_api_app, None)
        # Update the adjacency map based on the links
        self.update_adjacency(links_list)

    def update_adjacency(self, links):
        # Clear the adjacency map
        adjacency.clear()
        # Populate the adjacency map based on links
        for link in links:
            adjacency[link.src.dpid][link.dst.dpid] = link.src.port_no
            adjacency[link.dst.dpid][link.src.dpid] = link.dst.port_no

    def install_path(self, p, ev, src, dst):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        for i in range(0, len(p)-1):
            switch = p[i][0]
            in_port = p[i][1]
            out_port = p[i][2]
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            actions = [parser.OFPActionOutput(out_port)]
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0, command=ofproto.OFPFC_ADD,
                                    idle_timeout=0, hard_timeout=0, priority=2, instructions=inst)
            datapath.send_msg(mod)
