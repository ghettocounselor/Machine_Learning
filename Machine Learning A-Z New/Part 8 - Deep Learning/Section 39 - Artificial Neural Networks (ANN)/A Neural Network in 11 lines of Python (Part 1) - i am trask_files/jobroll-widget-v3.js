// This file will be minified by the closure compiler when you run ant deploy

window['indeedJobroll'] = (function() {
  // must maintain retry state on the client because the server JSP is dumb. It just sends back result html.
  // It doesn't know if we're retrying or not.
  var retry = false;

  var getJobs = function (offset) {
    var w = window;
    //extract last query from cookie
    var lq = (function(a) {
      var cookie = String(document.cookie), pos = cookie.indexOf(a + '=');
      if (pos == -1)
        return '';
      var d = cookie.indexOf(';', pos);
      var e = cookie.indexOf(':', pos);
      var c = cookie.substring(pos + a.length + 1, d == -1 ? (e == -1 ? cookie.length : e) : (e == -1 ? d : (e < d ? e : d)));
      var qs = c.indexOf("\"") == 0 ? 1 : 0;
      var qe = c.charAt(c.length - 1) == "\"" ? c.length - 1 : c.length;
      return c.substring(qs, qe);
    })("IND_RQ");
    // Send JSONP-style GET request by appending script
    var s = document.createElement("script")
    var h = document.getElementsByTagName("head")[0];
    s.async = true;
    s.src = ind_d +
        "/ads/jwidget.js?pub=" +
        encodeURIComponent(ind_pub) +
        (w.ind_chnl ? "&chnl=" +
            ind_chnl : "") +
        "&el=" + encodeURIComponent(ind_el) +
        (w.ind_pf ? "&pf=" + ind_pf : "") +
        (ind_q ? "&q=" + encodeURIComponent(ind_q) : "") +
        (w.ind_l ? "&l=" + encodeURIComponent(ind_l) : "") +
        (w.ind_n ? "&n=" + ind_n : "") +
        (lq ? "&lq=" + encodeURIComponent(lq) : "") +
        "&lm=" + (w.ind_t ? ind_t : "") + "+" +
        (w.ind_c ? ind_c : "") +
        (w.ind_nr || retry ? "&rtgt=0" : "") +
        //s is start offset. eg 0 on first page, 10 on second, 20 etc.
        (offset ? "&s=" + offset : "&s=0") +
        //ind_pgn is pagination toggle. set to 1 to activate.
        (w.ind_pgn ? "&pgn=" + w.ind_pgn : "") +
        (w.ind_pgnCnt ? "&pgnCnt=" + w.ind_pgnCnt : "") +
        (w.ind_age ? "&age=" + w.ind_age : "") +
        (w.ind_snp ? "&snp=" + w.ind_snp : "") +
        (w.ind_iaTxt ? "&iaTxt=" + w.ind_iaTxt : "") +
        "&v=3"; //version 3 of this script.
    h.appendChild(s);
    //prevent default behavior of clicking an anchor with href="#" that takes you to top of page
    return false;
  };


  var jobsCallback = function (elementId, jobsHtml) {
    var ind_jobs_el = document.getElementById(elementId);
    var ind_err = null;
    if (ind_jobs_el) {
      // if we didn't get any results, try again with the default query for this site
      if (jobsHtml.indexOf('class="job"') === -1 && retry === false) {
        //don't get caught in loop of queries if retry response is also empty
        retry = true;
        //try again but don't retarget
        window['indeedJobroll']['getJobs'](0);
      } else {
        ind_jobs_el.innerHTML = jobsHtml;
      }

    } else {
      ind_err = "no_elt";
    }
  };

  return {
    'getJobs': getJobs,
    'jobsCallback': jobsCallback
  };
})();
// get the first page of jobs on page load
window['indeedJobroll']['getJobs'](0);
