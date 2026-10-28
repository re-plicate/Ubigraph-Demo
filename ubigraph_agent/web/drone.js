/* Independent fleet stream: flight continues after a chat task ends. */
(() => {
  'use strict';
  const q = selector => document.querySelector(selector);
  const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colors = {'1':'#2463eb','2':'#078b85','3':'#c57920'};
  const labels = {idle:'Idle',ready:'Preparing',taking_off:'Take off',travelling:'Navigate to target',collecting:'Simulated collection',returning:'Return',landing:'Land',hovering:'Hover',running:'Running',completed:'Complete',cancelled:'Cancelled',cancelling:'Cancellation pending',failed:'Failed',fault:'Fault',interrupted:'Interrupted on restart',pending:'Pending'};
  const label = s => labels[s] || s;
  const clock = n => {n=Math.floor(n||0);return `${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;};
  let fleet = null, plan = null, selection = null, initialized = false, busy = false, geometry = null;
  let followLatest = true, menuKey = '';
  const terminal = m => ['completed','cancelled','failed','interrupted'].includes(m.status);
  const requestId = () => 'web-' + crypto.randomUUID().slice(0,18);

  async function request(path, body) {
    const response = await fetch('/api/drone/' + path, body === undefined ? {} : {
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Simulation request incomplete');
    return value;
  }

  async function action(button, fn) {
    if (busy) return;
    busy=true;button.disabled=true;
    try {await fn();} catch(error) {toast(error.message);} finally {busy=false;button.disabled=false;refreshButtons();}
  }

  function chosen() {return [...document.querySelectorAll('#drone-targets input:checked')].map(n=>n.value);}
  function clearPlan() {plan=null;q('#drone-plan-result').textContent='Inputs changed. Generate a new plan.';refreshButtons();draw();}
  function refreshButtons() {
    const active=!!fleet?.active_mission_id;
    const unavailable=!fleet || !!fleet.last_error;
    q('#drone-start').disabled=busy || active || unavailable || !plan?.feasible;
    q('#drone-plan').disabled=busy || active || unavailable;
    q('#drone-reset').disabled=busy || active;
    q('#drone-control').disabled=busy || active || unavailable;
    q('#drone-pause').disabled=busy || unavailable;
    q('#drone-speed').disabled=busy || unavailable;
    const m=fleet?.missions.find(m=>m.id===selection);
    q('#drone-cancel').disabled=busy || !m || terminal(m) || m.status==='cancelling' || m.status==='returning';
  }

  function render(value) {
    window.dispatchEvent(new CustomEvent("ubigraph-fleet-state",{detail:{revision:value.revision,epoch:value.epoch,missions:value.missions.map(m=>({id:m.id,plan_id:m.plan_id,status:m.status,reason:m.reason,completed:m.completed_targets.length,total:m.targets.length}))}}));
    fleet=value;
    if(plan && (plan.epoch!==value.epoch || plan.revision!==value.revision)){
      plan=null;q('#drone-plan-result').textContent='Fleet state changed. Generate a new plan.';
    }
    if (!initialized) {
      const nodes=Object.entries(value.scene.nodes);
      q('#drone-targets').innerHTML=nodes.map(([id,n])=>`<label title="${h(n.parking_alias||'Simulation grid')} · (${n.x}, ${n.y})"><input type="checkbox" value="${id}" ${['G1','G3','G5'].includes(id)?'checked':''}>${id}</label>`).join('');
      q('#drone-priority').innerHTML='<option value="">Equal priority</option>'+nodes.map(([id])=>`<option>${id}</option>`).join('');
      q('#drone-manual-target').innerHTML=nodes.map(([id])=>`<option>${id}</option>`).join('');
      document.querySelectorAll('#drone-targets input').forEach(input=>input.addEventListener('change',clearPlan));
      initialized=true;
    }
    q('#drone-clock').textContent=`Simulation ${clock(value.sim_time)}`;
    q('#drone-summary').textContent=value.last_error?'Simulator error. Reset the scene.':value.paused?'Paused':`Speed ${value.speed}× · ${value.active_mission_id?'Task running in the background':'Waiting for a task'}`;
    q('#drone-summary').title=value.last_error||'';
    q('#drone-stream-dot').style.background=value.last_error?'#bd655a':'#48a68b';
    q('#drone-pause').textContent=value.paused?'Resume':'Pause';
    q('#drone-speed').value=String(value.speed);
    q('#drone-fleet').innerHTML=value.fleet.map(d=>`<div class="drone-fleet-row"><div class="drone-fleet-title"><strong><i style="background:${colors[d.id]}"></i>Drone ${h(d.id)}</strong><span>${h(label(d.status))}</span></div><div class="drone-position">x ${d.x.toFixed(2)} · y ${d.y.toFixed(2)} · Altitude ${d.z.toFixed(2)} m</div><div class="drone-battery"><div><i style="width:${Math.max(0,d.battery)}%;${d.battery<20?'background:#c88a52':''}"></i></div><span>${d.battery.toFixed(1)}%</span></div></div>`).join('');
    if (followLatest || !selection || !value.missions.some(m=>m.id===selection)) selection=value.active_mission_id||value.missions.at(-1)?.id||null;
    const select=q('#drone-mission-select');
    const key=value.missions.map(m=>m.id+'|'+m.status).join(',');
    if(key!==menuKey || !select.options.length){
      select.innerHTML=value.missions.length?value.missions.slice().reverse().map(m=>`<option value="${h(m.id)}">${h(label(m.status))} · ${h(m.id.slice(-8))}</option>`).join(''):'<option>No tasks yet</option>';
      menuKey=key;
    }
    if(selection)select.value=selection;
    if(!plan && value.missions.length){const latest=value.missions.at(-1);q('#drone-plan-result').textContent=`Latest mission: ${label(latest.status)} · ${latest.reason}`;}
    renderMission();renderEvents();refreshButtons();draw();
  }

  function renderMission() {
    const m=fleet?.missions.find(m=>m.id===selection);
    if(!m){q('#drone-mission').innerHTML='<p class="muted small">Targets are provided externally. Select nodes, generate a plan, then execute.</p>';return;}
    const routes=m.original_routes||m.routes;
    const targets=m.targets.length?m.targets:['Drone controls'];
    const elapsed=(m.finished_sim_time??fleet.sim_time)-m.started_sim_time;
    q('#drone-mission').innerHTML=`<p class="small drone-status-${h(m.status)}"><strong>${h(label(m.status))}</strong>${m.epoch!==fleet.epoch?' · Replay scenario task':''} · ${h(m.reason)}</p><div class="drone-mission-metrics"><div><strong>${m.completed_targets.length} / ${m.targets.length}</strong><small>Collection targets completed</small></div><div><strong>${elapsed.toFixed(1)}s</strong><small>Actual simulation time</small></div><div><strong>${m.distance_m.toFixed(1)}m</strong><small>Actual total distance</small></div></div><div class="table-scroll"><table class="drone-receipts"><thead><tr><th>Target / Action</th><th>Drone</th><th>Status</th><th>Receipt</th></tr></thead><tbody>${targets.map(target=>{
      const pair=Object.entries(routes).find(([,r])=>r.targets.includes(target));
      const job=pair?.[1].jobs.find(j=>j.node===target),receipt=m.receipts.find(r=>r.node===target);
      const status=receipt?'completed':(job?.status||(!terminal(m)?'pending':m.status));
      return `<tr><td>${h(target)}</td><td>${h(pair?.[0]||Object.keys(m.routes).join(', '))}</td><td>${h(label(status))}</td><td>${receipt?`Simulation ${clock(receipt.completed_sim_time)} · Action receipt`: '—'}</td></tr>`;
    }).join('')}</tbody></table></div><p class="card-note">Task ${h(m.id)}<br>Collection receipts contain no measured parking availability and do not trigger forecasting or training.</p>`;
  }

  function renderEvents() {
    const events=fleet.events.filter(e=>!selection||e.mission_id===selection||!e.mission_id).slice(-24).reverse();
    q('#drone-events').innerHTML=events.length?events.map(e=>{
      let text=e.message.replace(/(\d+)\s*entered\s*(.+?)\s*Stage/g,'Drone $1 · $2').replace(/(\d+)\s*reached\s*(G\d+)[，, ]+Start simulated collection/g,'Drone $1 reached $2 · Collection started').replace(/(G\d+)\s*Simulated collection completed/g,'$1 · Collection completed');Object.entries(labels).forEach(([a,b])=>{text=text.replace(a,b);});
      return `<div class="drone-event"><time>${clock(e.sim_time)}</time><span>${h(text)}</span></div>`;
    }).join(''):'<p class="muted small">Waiting for execution events.</p>';
  }

  function draw() {
    if(!fleet||q('#drone-view').classList.contains('hidden'))return;
    const canvas=q('#drone-canvas'),w=canvas.clientWidth,ht=w*.8,dpr=window.devicePixelRatio||1;
    if(!w)return;
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(ht*dpr);
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
    const pad=20,unit=Math.min((w-pad*2)/20,(ht-pad*2)/16);
    const point=(x,y)=>[pad+(x+.5)*unit,ht-pad-(y+.5)*unit];
    geometry={pad,unit,ht};
    ctx.fillStyle='#fafbfd';ctx.fillRect(0,0,w,ht);
    ctx.strokeStyle='#e9eef5';ctx.lineWidth=.6;
    for(let x=0;x<=20;x++){ctx.beginPath();ctx.moveTo(pad+x*unit,pad);ctx.lineTo(pad+x*unit,ht-pad);ctx.stroke();}
    for(let y=0;y<=16;y++){ctx.beginPath();ctx.moveTo(pad,ht-pad-y*unit);ctx.lineTo(pad+20*unit,ht-pad-y*unit);ctx.stroke();}
    ctx.font='9px system-ui';ctx.fillStyle='#94a1b4';ctx.textAlign='center';
    for(let x=0;x<20;x+=5){const [px]=point(x,0);ctx.fillText(x+'m',px,ht-4);}
    fleet.scene.obstacles.forEach(([x,y])=>{const [px,py]=point(x,y);ctx.fillStyle='#cbd5e2';ctx.fillRect(px-unit*.45,py-unit*.45,unit*.9,unit*.9);});
    const mission=fleet.missions.find(m=>m.id===selection);
    const routes=(!fleet.active_mission_id&&plan?.routes)||mission?.routes||{};
    Object.entries(routes).forEach(([id,r])=>{if(!r.path)return;ctx.beginPath();r.path.forEach((p,i)=>{const [x,y]=point(p.x,p.y);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.strokeStyle=colors[id]+'75';ctx.lineWidth=1.7;ctx.setLineDash([4,5]);ctx.stroke();ctx.setLineDash([]);});
    Object.entries(fleet.scene.homes).forEach(([id,[x,y]])=>{const [px,py]=point(x,y);ctx.beginPath();ctx.arc(px,py,unit*.38,0,2*Math.PI);ctx.strokeStyle=colors[id]+'70';ctx.lineWidth=1.3;ctx.setLineDash([2,3]);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=colors[id];ctx.font='9px system-ui';ctx.fillText('Base'+id,px,py+unit*.9);});
    const selected=new Set(chosen());
    Object.entries(fleet.scene.nodes).forEach(([id,n])=>{const [x,y]=point(n.x,n.y),done=mission?.completed_targets.includes(id);ctx.beginPath();ctx.arc(x,y,Math.max(5,unit*.2),0,2*Math.PI);ctx.fillStyle=done?'#d7eee4':selected.has(id)?'#dfebff':'#eff3f8';ctx.fill();ctx.strokeStyle=done?'#68a48c':selected.has(id)?'#739ce6':'#b6c3d4';ctx.stroke();ctx.fillStyle=done?'#49846d':'#576f91';ctx.font='bold 10px system-ui';ctx.fillText(done?'✓':id,x,y-10);});
    fleet.fleet.forEach(d=>{
      ctx.beginPath();d.trail.forEach(([x,y],i)=>{const p=point(x,y);i?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.strokeStyle=colors[d.id]+'b0';ctx.lineWidth=2;ctx.stroke();
      const [x,y]=point(d.x,d.y);ctx.save();ctx.translate(x,y);ctx.rotate(-d.yaw*Math.PI/180);
      ctx.strokeStyle=colors[d.id];ctx.fillStyle='white';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(-5,-5);ctx.lineTo(5,5);ctx.moveTo(-5,5);ctx.lineTo(5,-5);ctx.stroke();
      for(const [a,b] of [[-5,-5],[-5,5],[5,-5],[5,5]]){ctx.beginPath();ctx.arc(a,b,3.3,0,2*Math.PI);ctx.fill();ctx.stroke();}
      ctx.beginPath();ctx.arc(0,0,3,0,2*Math.PI);ctx.fillStyle=d.fault?'#bd655a':colors[d.id];ctx.fill();ctx.restore();
      ctx.fillStyle=colors[d.id];ctx.font='bold 9px system-ui';ctx.fillText(d.id+'No.',x+15,y+3);
    });
  }

  q('#drone-plan').onclick=()=>action(q('#drone-plan'),async()=>{
    const targets=chosen(),priority=q('#drone-priority').value;
    plan=await request('plan',{request_id:requestId(),targets,drone_ids:[...document.querySelectorAll('.drone-ids input:checked')].map(n=>n.value),priority_targets:priority?[priority]:[],dwell_seconds:Number(q('#drone-dwell').value),return_to_base:q('#drone-return').checked});
    showPlan(plan);
  });
  function showPlan(p) {
    if(p.command_manifest)window.dispatchEvent(new CustomEvent("ubigraph-command-manifest",{detail:p.command_manifest}));
    q('#drone-plan-result').innerHTML=p.feasible?`<strong>Feasible plan · Estimated ${p.duration_seconds.toFixed(0)} seconds</strong><br>${Object.entries(p.routes).map(([id,r])=>`${h(id)}No. → ${r.targets.map(h).join(' → ')}`).join('<br>')}<br>Estimated horizontal distance ${p.distance_m.toFixed(1)}m；Not executed yet.<br>Collection dwell rounds up to 2 -second time slots (rounded up).`:`<strong>Infeasible plan</strong><br>${p.rejected.map(r=>h(r.target+'：'+r.reason)).join('<br>')}`;
    refreshButtons();draw();
  }
  q('#drone-start').onclick=()=>action(q('#drone-start'),async()=>{const m=await request('start',{plan_id:plan.id});selection=m.id;followLatest=true;plan=null;q('#drone-plan-result').textContent='Mission submitted. Progress and collection receipts update live.';toast('Mission submitted. Track live flight progress.');render(await request('state'));window.dispatchEvent(new CustomEvent("ubigraph-drone-artifact",{detail:m}));});
  q('#drone-cancel').onclick=()=>action(q('#drone-cancel'),async()=>{await request('cancel',{mission_id:selection});toast('Cancellation requested. Drones will return after the current segment.');render(await request('state'));});
  q('#drone-pause').onclick=()=>action(q('#drone-pause'),async()=>render(await request('settings',{paused:!fleet.paused})));
  q('#drone-speed').onchange=()=>action(q('#drone-speed'),async()=>render(await request('settings',{speed:Number(q('#drone-speed').value)})));
  q('#drone-reset').onclick=()=>action(q('#drone-reset'),async()=>{plan=null;selection=null;followLatest=true;render(await request('reset',{}));q('#drone-plan-result').textContent='Scene reset. Previous mission records are preserved.';});
  q('#drone-control').onclick=()=>action(q('#drone-control'),async()=>{
    const act=q('#drone-manual-action').value;
    const m=await request('control',{request_id:requestId(),drone_id:q('#drone-manual-id').value,action:act,target:act==='navigate'?q('#drone-manual-target').value:'NONE',height:.6,duration_seconds:3});
    selection=m.id;followLatest=true;plan=null;render(await request('state'));toast('Simulation action submitted.');
  });
  for(const [id,params] of [['#drone-low-battery',{battery:18}],['#drone-fault',{fault:true}],['#drone-recover',{fault:false,battery:100}]])q(id).onclick=()=>action(q(id),async()=>{clearPlan();render(await request('configure',{drone_id:q('#drone-manual-id').value,...params}));});
  q('#drone-mission-select').onchange=()=>{selection=q('#drone-mission-select').value;followLatest=false;renderMission();renderEvents();refreshButtons();draw();};
  for(const id of ['#drone-priority','#drone-dwell','#drone-return'])q(id).addEventListener('change',clearPlan);
  document.querySelectorAll('.drone-ids input').forEach(input=>input.addEventListener('change',clearPlan));
  document.querySelectorAll('[data-drone-prompt]').forEach(b=>b.onclick=()=>send(b.dataset.dronePrompt));
  q('#drone-canvas').onclick=event=>{
    if(!geometry||!fleet)return;const r=event.currentTarget.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;
    const {pad,unit,ht}=geometry;
    const found=Object.entries(fleet.scene.nodes).find(([,n])=>Math.hypot(x-(pad+(n.x+.5)*unit),y-(ht-pad-(n.y+.5)*unit))<unit*.7);
    if(found){const input=q(`#drone-targets input[value="${found[0]}"]`);input.checked=!input.checked;clearPlan();}
  };
  window.addEventListener('resize',draw);
  window.addEventListener('ubigraph-view',draw);
  window.addEventListener('ubigraph-drone-artifact',event=>{
    const a=event.detail;
    if(a.kind==='drone_mission'){selection=a.id;followLatest=true;plan=null;q('#drone-plan-result').textContent=a.reason||'AgentMission submitted. Check actual progress.';}
    if(a.kind==='drone_plan'){plan=a;showPlan(a);}
    request('state').then(render).catch(()=>{});
  });
  let retry=null;
  const stream=new EventSource('/api/drone/events');
  stream.onmessage=event=>{if(retry){clearInterval(retry);retry=null;}q('#drone-stream-dot').style.background='#48a68b';try{render(JSON.parse(event.data));}catch(error){toast('Unable to display simulation state:'+error.message);}};
  stream.onerror=()=>{q('#drone-stream-dot').style.background='#d5a25d';q('#drone-summary').textContent='Reconnecting to simulator...';if(!retry)retry=setInterval(()=>request('state').then(render).catch(()=>{}),2500);};
  window.addEventListener('pagehide',()=>{stream.close();if(retry)clearInterval(retry);});
})();
