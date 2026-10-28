/* Product surfaces read the same engine, sessions and simulator as chat. */
(() => {
  'use strict';
  let operations=null, pending=false, commandPlan=null, commandRevision=null;
  const statusLabels={ready:'Ready',unconfigured:'Configuration needed',experimental:'Research baseline',unavailable:'Unavailable'};
  const taskLabels={completed:'Completed',running:'Running',queued:'Queued',failed:'Incomplete',cancelled:'Stopped',interrupted:'Interrupted'};
  const flightLabels={idle:'Idle',taking_off:'Taking off',travelling:'Navigating',collecting:'Collecting',returning:'Returning',landing:'Landing',hovering:'Hover',fault:'Fault'};
  const toolAgent={inspect_history:'data',predict_parking:'temporal',get_spatial_similarity:'spatial',explain_prediction:'temporal',assess_collection_need:'hic',get_fleet_status:'drone',plan_collection_mission:'drone',start_collection_mission:'drone',get_mission_status:'drone',cancel_mission:'drone',control_drone:'drone',get_detection_status:'detection',transfer_parking_estimate:'transfer'};
  function selectAgent(id){const select=$('#active-agent');if([...select.options].some(o=>o.value===id))select.value=id;$('#active-agent-hint').textContent=operations?.agents.find(a=>a.id===id)?.description||'Coordinated specialist tools';}
  function openChat(text=null,agent='orchestrator'){
    selectAgent(agent);view('chat');
    if(text)send(text);else $('#question').focus();
  }
  function drawCity(){
    const a=state.bootstrap?.spatial;if(!a?.coordinates)return;
    const points=a.coordinates,lat=points.map(p=>p.latitude),lon=points.map(p=>p.longitude);
    const minLat=Math.min(...lat),maxLat=Math.max(...lat),minLon=Math.min(...lon),maxLon=Math.max(...lon);
    const xy=p=>[85+425*(p.longitude-minLon)/(maxLon-minLon||1),275-200*(p.latitude-minLat)/(maxLat-minLat||1)];
    let blocks='',roads='',links='',nodes='';
    for(let i=0;i<12;i++)for(let j=0;j<7;j++){
      if((i*3+j*5)%7<2)continue;
      const x=15+i*52+j*4,y=12+j*47,skew=(i%3)*4;
      blocks+=`<path d="M${x} ${y}h${30-skew}v${25+skew}h-${30-skew}Z" fill="${(i+j)%3===0?'#244b3d':'#1c4235'}" stroke="#2d5543" stroke-width=".6"/>`;
    }
    for(let i=0;i<7;i++)roads+=`<path d="M0 ${37+i*47}L600 ${-5+i*47}" stroke="#3b6149" stroke-width="1.4"/>`;
    for(let i=0;i<12;i++)roads+=`<path d="M${i*53} 0L${i*53+45} 350" stroke="#315a42" stroke-width="1"/>`;
    for(let i=0;i<points.length;i++){
      const [x,y]=xy(points[i]);
      for(let j=i+1;j<points.length;j++){
        if(a.matrix[i][j]<.25)continue;const [x2,y2]=xy(points[j]);
        links+=`<path d="M${x} ${y} Q${(x+x2)/2+25} ${(y+y2)/2-20} ${x2} ${y2}" fill="none" stroke="#add18c" opacity="${.2+a.matrix[i][j]*.25}" stroke-width="1" stroke-dasharray="3 5"/>`;
      }
      nodes+=`<g><circle cx="${x}" cy="${y}" r="14" fill="#bfe6a2" opacity=".08"/><circle cx="${x}" cy="${y}" r="8" fill="#c6e6ab" opacity=".16"/><circle cx="${x}" cy="${y}" r="3.4" fill="#ccefae" stroke="#ecffcf" stroke-width="1"/><text x="${x+12}" y="${y+4}" fill="#c0d6ad" font-size="7.5" font-family="monospace">${esc(points[i].garage)}</text></g>`;
    }
    $('#city-map').innerHTML=`<defs><pattern id="map-dot" width="10" height="10" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".4" fill="#365d45"/></pattern></defs><rect width="600" height="350" fill="url(#map-dot)"/>${blocks}${roads}<path d="M600 60C450 140 535 220 445 350" stroke="#294e40" stroke-width="28" fill="none"/>${links}${nodes}<text x="25" y="315" fill="#658d70" font-size="7" font-family="monospace">56°09′ N  /  10°12′ E</text>`;
    $('#map-node-count').textContent=points.length;
  }
  function renderAgents(rows){
    const selected=$('#active-agent').value;
    $('#active-agent').innerHTML=rows.map(a=>`<option value="${esc(a.id)}">${esc(a.name)}${a.id==='orchestrator'?' · Automatic routing':''}</option>`).join('');
    selectAgent(selected||'orchestrator');
    $('#agent-directory').innerHTML=rows.map(a=>`<article class="agent-profile" data-agent-card="${esc(a.id)}"><div class="agent-profile-head"><span class="agent-avatar">${esc(a.initials)}</span><span class="agent-state ${esc(a.status)}"><i></i>${esc(statusLabels[a.status]||a.status)}</span></div><h2>${esc(a.name)}</h2><p>${esc(a.description.replace('Use the trained V3 model computes next-step availability','Compute next-step availability with the trained V3 model'))}</p><button data-agent-chat="${esc(a.id)}">${a.status==='unconfigured'?'Configuration requirements':'Chat with agent'} <span>↗</span></button></article>`).join('');
    document.querySelectorAll('[data-agent-chat]').forEach(b=>b.onclick=()=>{
      if(b.dataset.agentChat==='detection'&&rows.find(a=>a.id==='detection')?.status==='unconfigured')openChat('Check vision detection Agent actual configuration status and explain what is required.','detection');
      else openChat(null,b.dataset.agentChat);
    });
  }
  function renderOperations(data){
    operations=data;
    $('#stat-agents').textContent=data.agents.filter(a=>a.status==='ready').length;
    $('#stat-nodes').textContent=data.overview.items.filter(a=>a.valid).length+' / '+data.overview.items.length;
    $('#stat-fleet').textContent=data.fleet.fleet.filter(d=>d.connected&&!d.fault&&d.status==='idle'&&d.battery>=15).length+' / '+data.fleet.fleet.length;
    $('#stat-tasks').textContent=num(data.activity.completed,0);
    $('#overview-activity').innerHTML=data.activity.recent.length?data.activity.recent.map(t=>`<div class="activity-row"><span class="activity-dot">${t.status==='completed'?'✓':t.status==='failed'?'!':'◈'}</span><div><strong>${esc(displayTitle(t.title))}</strong><small>${esc(new Date(t.created*1000).toLocaleString('en-US',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}))}</small></div><span>${esc(taskLabels[t.status]||t.status)}</span></div>`).join(''):'<div class="product-empty">No collaboration yet. Ask the team a question to start your first task.</div>';
    $('#overview-fleet').innerHTML=data.fleet.fleet.map(d=>`<div class="preview-drone"><span>✣</span><div><strong>DRONE 0${esc(d.id)}</strong><small>${esc(flightLabels[d.status]||d.status)} · ${d.fault?'Fault':d.connected?'Simulator connected':'Disconnected'}</small></div><div class="battery-mini"><b>${num(d.battery,0)}%</b><i><em style="width:${Math.max(0,Math.min(100,d.battery))}%"></em></i></div></div>`).join('');
    renderAgents(data.agents);drawCity();
  }
  async function refreshOperations(){
    if(pending)return;pending=true;
    try{const q=new URLSearchParams(state.context||{});renderOperations(await api('/api/operations?'+q));}
    catch(e){$('#overview-activity').innerHTML='<div class="product-error">Unable to update workspace:'+esc(e.message)+'</div>';$('#agent-directory').innerHTML='<div class="product-error">Unable to load team status. Refresh the workspace and try again.</div>';}
    finally{pending=false;}
  }
  const actionNames={TAKEOFF:'Take off',NAVIGATE:'Navigate to target',COLLECT:'Simulated collection',RETURN_TO_BASE:'Return to base',LAND:'Land',HOLD_POSITION:'Hold position'};
  async function renderCommands(value){
    commandPlan=value.plan_id;
    $('.command-center').classList.toggle('is-executed',!!value.started);
    try{localStorage.setItem('ubigraph-command-plan',commandPlan);}catch{}
    const groups=new Map();for(const c of value.commands){if(!groups.has(c.drone_id))groups.set(c.drone_id,[]);groups.get(c.drone_id).push(c);}
    const disabled=value.stale||value.started||!value.feasible;
    $('#flight-commands').innerHTML=`<div class="command-grid">${[...groups].map(([id,commands])=>`<article class="command-vehicle"><div class="command-vehicle-header"><strong>DRONE 0${esc(id)}</strong><small>${commands.length}  actions</small></div>${commands.map(c=>`<div class="command-step"><span>${String(c.sequence).padStart(2,'0')}</span><div><strong>${esc(actionNames[c.action]||c.action)}${c.target?' · '+esc(c.target):''}</strong><small>${c.position_m?'Position '+c.position_m.map(v=>num(v,1)).join(' / ')+' m':c.height_m!==undefined?'Altitude '+num(c.height_m,1)+' m':c.duration_seconds!==undefined?num(c.duration_seconds,1)+'  seconds · Action receipt':''}</small></div></div>`).join('')}</article>`).join('')}</div><div class="command-actions"><div class="command-meta">${esc(value.plan_id)}<br>${value.started?'Plan already submitted. Follow progress in mission receipts.':value.stale?'Fleet state changed. This plan expired; generate a new plan.':!value.feasible?'Plan is infeasible and cannot execute.':'Plan generated. No takeoff yet. Fleet state and return battery are rechecked at execution.'}</div><button id="execute-command-plan" class="primary-button" ${disabled?'disabled':''}>${value.started?'Submitted':'Execute simulation commands →'}</button></div><details class="raw-details"><summary>View structured commands JSON</summary><pre>${esc(JSON.stringify(value,null,2))}</pre></details>`;
    const download=document.createElement('button');download.className='text-action';download.textContent='Download commands JSON ↓';
    download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=value.plan_id+'-commands.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    $('#flight-commands').append(download);
    $('#execute-command-plan').onclick=async()=>{
      const b=$('#execute-command-plan');b.disabled=true;
      try{const mission=await api('/api/drone/start',{plan_id:commandPlan});window.dispatchEvent(new CustomEvent('ubigraph-drone-artifact',{detail:mission}));toast('Simulation mission submitted. Live status will update.');renderCommands(await api('/api/drone/plans/'+commandPlan+'/commands'));}
      catch(e){toast(e.message);b.disabled=false;}
    };
  }
  async function loadCommands(id){try{await renderCommands(await api('/api/drone/plans/'+encodeURIComponent(id)+'/commands'));}catch{}}
  $('#product-settings').onclick=()=>openSettings();
  $('#hero-chat').onclick=()=>openChat();$('#refresh-operations').onclick=refreshOperations;
  $('#active-agent').onchange=()=>selectAgent($('#active-agent').value);
  document.querySelectorAll('[data-product-prompt]').forEach(b=>b.onclick=()=>openChat(b.dataset.productPrompt));
  window.addEventListener('ubigraph-ready',()=>{refreshOperations();drawCity();});
  window.addEventListener('ubigraph-view',event=>{if(['overview','agents'].includes(event.detail))refreshOperations();if(event.detail==='drone'&&commandPlan)loadCommands(commandPlan);});
  window.addEventListener('ubigraph-agent-event',({detail:{event,replay}})=>{
    const d=event.data||{};
    if(event.type==='tool_started'&&!replay){const id=toolAgent[d.name];document.querySelectorAll('[data-agent-card]').forEach(n=>n.classList.toggle('is-working',n.dataset.agentCard===id));}
    if(event.type==='artifact'&&d.kind==='drone_plan'&&d.command_manifest)renderCommands(d.command_manifest);
    if(['done','error','cancelled'].includes(event.type)&&!replay){refreshOperations();document.querySelectorAll('.is-working').forEach(n=>n.classList.remove('is-working'));}
  });
  window.addEventListener('ubigraph-fleet-state',({detail})=>{for(const m of detail.missions||[]){const card=document.querySelector('[data-mission-card="'+m.id+'"]');if(card)card.querySelector('.mission-live-text').textContent=m.reason+' · '+m.completed+'/'+m.total;const planCard=document.querySelector('[data-plan-card="'+m.plan_id+'"]');if(planCard)planCard.querySelector('.mission-live-text').textContent='Plan submitted · '+m.reason;}const key=detail.epoch+'|'+detail.revision;if(key!==commandRevision){commandRevision=key;if(commandPlan)loadCommands(commandPlan);}});
  window.addEventListener('ubigraph-command-manifest',({detail})=>renderCommands(detail));
  window.addEventListener('ubigraph-drone-artifact',({detail:a})=>{if(a.kind==='drone_plan'&&a.command_manifest)renderCommands(a.command_manifest);if(a.kind==='drone_mission'&&a.plan_id)loadCommands(a.plan_id);});
  try{commandPlan=localStorage.getItem('ubigraph-command-plan');if(commandPlan)loadCommands(commandPlan);}catch{}
  let timer=setInterval(()=>{if(!document.hidden&&['overview','agents'].includes(state.view))refreshOperations();},15000);
  window.addEventListener('pagehide',()=>clearInterval(timer));
  refreshOperations();
})();
