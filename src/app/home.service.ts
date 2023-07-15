import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class HomeService {
  
  type='me';
  nodeURL = 'http://localhost:3000/';
  constructor(private http: HttpClient,) { }

  homeApi(data:any){
    let httpParams = new HttpParams().set("query", data.query);
    if(data.startTime?.length){
      httpParams = httpParams.set("startTime", data.startTime.toISOString());
    }
    if(data.endTime?.length){
      httpParams = httpParams.set("endTime", data.endTime.toISOString());
    }
    httpParams = httpParams.set("max", data.max);
    httpParams = httpParams.set("Content-Type", 'application/json');
    return this.http.get<any>(this.nodeURL, {params:httpParams});
  }
}
